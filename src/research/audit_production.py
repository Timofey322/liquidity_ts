from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.validate import validate_ohlcv
from src.fast import detect_fvg_zones, fractal_highs, fractal_lows
from src.research.build_production_portfolio import PRODUCTION_CONFIGS
from src.research.mtf_reversal import (
    ResearchConfig,
    _build_htf_context,
    _confirmed_flags,
    _cross_series,
    _in_last_fvg_zone,
)


def _simulate_side_with_risk(df: pd.DataFrame, idx: int, side: str, cfg: ResearchConfig) -> tuple[float, int, float]:
    entry = float(df["close"].iloc[idx])
    atr = float(df["atr5"].iloc[idx])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return 0.0, 1, np.nan

    last_fh = float(df["htf_last_fh"].iloc[idx]) if np.isfinite(df["htf_last_fh"].iloc[idx]) else np.nan
    last_fl = float(df["htf_last_fl"].iloc[idx]) if np.isfinite(df["htf_last_fl"].iloc[idx]) else np.nan
    sweep_lo = float(df["htf_sweep_long_level"].iloc[idx]) if np.isfinite(df["htf_sweep_long_level"].iloc[idx]) else np.nan
    sweep_hi = float(df["htf_sweep_short_level"].iloc[idx]) if np.isfinite(df["htf_sweep_short_level"].iloc[idx]) else np.nan

    if side == "long":
        stop_candidates = [entry - float(cfg.stop_atr_mult) * atr]
        if np.isfinite(last_fl) and last_fl < entry:
            stop_candidates.append(last_fl)
        if np.isfinite(sweep_lo) and sweep_lo < entry:
            stop_candidates.append(sweep_lo)
        stop = min(stop_candidates)
        risk = entry - stop
        if risk <= 1e-12:
            return 0.0, 1, np.nan
        target_candidates = []
        if np.isfinite(last_fh) and last_fh > entry:
            target_candidates.append(last_fh)
        target = min(target_candidates) if target_candidates else entry + float(cfg.target_rr) * risk
    else:
        stop_candidates = [entry + float(cfg.stop_atr_mult) * atr]
        if np.isfinite(last_fh) and last_fh > entry:
            stop_candidates.append(last_fh)
        if np.isfinite(sweep_hi) and sweep_hi > entry:
            stop_candidates.append(sweep_hi)
        stop = max(stop_candidates)
        risk = stop - entry
        if risk <= 1e-12:
            return 0.0, 1, np.nan
        target_candidates = []
        if np.isfinite(last_fl) and last_fl < entry:
            target_candidates.append(last_fl)
        target = max(target_candidates) if target_candidates else entry - float(cfg.target_rr) * risk

    last_idx = min(idx + int(cfg.max_hold_bars), len(df) - 1)
    exit_px = float(df["close"].iloc[last_idx])
    hold = last_idx - idx
    for j in range(idx + 1, last_idx + 1):
        lo = float(df["low"].iloc[j])
        hi = float(df["high"].iloc[j])
        if side == "long":
            if lo <= stop:
                exit_px = stop
                hold = j - idx
                break
            if hi >= target:
                exit_px = target
                hold = j - idx
                break
        else:
            if hi >= stop:
                exit_px = stop
                hold = j - idx
                break
            if lo <= target:
                exit_px = target
                hold = j - idx
                break

    if side == "long":
        ret = (exit_px / entry) - 1.0
    else:
        ret = 1.0 - (exit_px / entry)
    risk_pct = risk / entry
    return float(ret), int(max(hold, 1)), float(risk_pct)


def _build_audit_dataset(base: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    df = base.copy().sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)

    fh = fractal_highs(high, cfg.fractal_left, cfg.fractal_right)
    fl = fractal_lows(low, cfg.fractal_left, cfg.fractal_right)
    fh_flag, fh_val = _confirmed_flags(fh, high, cfg.fractal_right)
    fl_flag, fl_val = _confirmed_flags(fl, low, cfg.fractal_right)
    bull, bear, z_lo, z_hi = detect_fvg_zones(high, low, close)
    in_bull_fvg, in_bear_fvg, fvg_w = _in_last_fvg_zone(close, bull, bear, z_lo, z_hi)
    cross_up, cross_dn = _cross_series(high, low, close, fh, fl, cfg.fractal_right)

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()
    df["atr5"] = atr
    df["ret1"] = pd.Series(close).pct_change(fill_method=None).fillna(0.0)
    df["ret3"] = pd.Series(close).pct_change(3, fill_method=None).fillna(0.0)
    df["ret6"] = pd.Series(close).pct_change(6, fill_method=None).fillna(0.0)
    df["atr_rel"] = atr / (close + 1e-12)
    df["body_rel"] = (close - open_) / (atr + 1e-12)
    df["vol_log1p"] = np.log1p(np.maximum(vol, 0.0))
    df["hour_utc"] = df["ts"].dt.hour
    df["footprint_raw"] = df["footprint_raw"].astype(np.float64)
    fp_mu = df["footprint_raw"].rolling(cfg.footprint_z_window, min_periods=5).mean()
    fp_sd = df["footprint_raw"].rolling(cfg.footprint_z_window, min_periods=5).std(ddof=0)
    df["footprint_z"] = ((df["footprint_raw"] - fp_mu) / (fp_sd + 1e-12)).fillna(0.0)
    df["fh"] = fh_flag.astype(np.int8)
    df["fl"] = fl_flag.astype(np.int8)
    df["cross_up"] = cross_up.astype(np.int8)
    df["cross_dn"] = cross_dn.astype(np.int8)
    df["in_bull_fvg"] = in_bull_fvg.astype(np.int8)
    df["in_bear_fvg"] = in_bear_fvg.astype(np.int8)
    df["fvg_width"] = fvg_w / (close + 1e-12)

    htf = _build_htf_context(df, cfg)
    df = pd.merge_asof(df.sort_values("ts"), htf.sort_values("ts"), on="ts", direction="backward")
    base_step_sec = int(pd.Timedelta(cfg.base_rule).total_seconds())
    age_long = ((df["ts"] - pd.to_datetime(df["htf_last_long_idea_ts"], utc=True)).dt.total_seconds() / base_step_sec).fillna(1e9)
    age_short = ((df["ts"] - pd.to_datetime(df["htf_last_short_idea_ts"], utc=True)).dt.total_seconds() / base_step_sec).fillna(1e9)
    df["htf_age_long"] = age_long
    df["htf_age_short"] = age_short
    df["htf_long_bias"] = ((age_long >= 0) & (age_long <= cfg.idea_window_bars) & (age_long <= age_short)).astype(np.int8)
    df["htf_short_bias"] = ((age_short >= 0) & (age_short <= cfg.idea_window_bars) & (age_short < age_long)).astype(np.int8)

    session_ok = np.ones(len(df), dtype=bool)
    if cfg.session_start_hour_utc is not None and cfg.session_end_hour_utc is not None:
        start_h = int(cfg.session_start_hour_utc)
        end_h = int(cfg.session_end_hour_utc)
        if start_h <= end_h:
            session_ok = (df["hour_utc"] >= start_h) & (df["hour_utc"] < end_h)
        else:
            session_ok = (df["hour_utc"] >= start_h) | (df["hour_utc"] < end_h)
    df["session_ok"] = session_ok.astype(np.int8)

    vol_ok = np.ones(len(df), dtype=bool)
    q_window = int(cfg.atr_quantile_window)
    min_q = float(cfg.min_atr_quantile)
    max_q = float(cfg.max_atr_quantile)
    if q_window > 1 and (min_q > 0.0 or max_q < 1.0):
        min_periods = max(50, q_window // 4)
        atr_hist = df["atr_rel"].shift(1)
        q_lo = atr_hist.rolling(q_window, min_periods=min_periods).quantile(min_q) if min_q > 0.0 else pd.Series(-np.inf, index=df.index)
        q_hi = atr_hist.rolling(q_window, min_periods=min_periods).quantile(max_q) if max_q < 1.0 else pd.Series(np.inf, index=df.index)
        vol_ok = (df["atr_rel"] >= q_lo.fillna(-np.inf)) & (df["atr_rel"] <= q_hi.fillna(np.inf))
    df["vol_regime_ok"] = vol_ok.astype(np.int8)

    fp_thr = float(cfg.footprint_threshold)
    if cfg.entry_mode == "cross_only":
        long_confirm = (df["footprint_z"] > fp_thr) & (df["cross_up"] == 1)
        short_confirm = (df["footprint_z"] < -fp_thr) & (df["cross_dn"] == 1)
    elif cfg.entry_mode == "cross_and_ret":
        long_confirm = (df["footprint_z"] > fp_thr) & (df["cross_up"] == 1) & (df["ret1"] > 0)
        short_confirm = (df["footprint_z"] < -fp_thr) & (df["cross_dn"] == 1) & (df["ret1"] < 0)
    else:
        long_confirm = (df["footprint_z"] > fp_thr) & ((df["cross_up"] == 1) | (df["ret1"] > 0))
        short_confirm = (df["footprint_z"] < -fp_thr) & ((df["cross_dn"] == 1) | (df["ret1"] < 0))
    df["long_setup"] = (df["htf_long_bias"] == 1) & long_confirm & session_ok & vol_ok
    df["short_setup"] = (df["htf_short_bias"] == 1) & short_confirm & session_ok & vol_ok

    df["long_ret"] = 0.0
    df["short_ret"] = 0.0
    df["long_hold_bars"] = 1
    df["short_hold_bars"] = 1
    df["long_risk_pct"] = np.nan
    df["short_risk_pct"] = np.nan
    for i in range(len(df) - 1):
        if bool(df["long_setup"].iloc[i]):
            r, h, rp = _simulate_side_with_risk(df, i, "long", cfg)
            df.loc[i, ["long_ret", "long_hold_bars", "long_risk_pct"]] = [r, h, rp]
        if bool(df["short_setup"].iloc[i]):
            r, h, rp = _simulate_side_with_risk(df, i, "short", cfg)
            df.loc[i, ["short_ret", "short_hold_bars", "short_risk_pct"]] = [r, h, rp]

    keep = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "ret1",
        "ret3",
        "ret6",
        "atr_rel",
        "body_rel",
        "vol_log1p",
        "footprint_raw",
        "footprint_z",
        "fh",
        "fl",
        "cross_up",
        "cross_dn",
        "in_bull_fvg",
        "in_bear_fvg",
        "fvg_width",
        "htf_ret1",
        "htf_atr_rel",
        "htf_vol_log1p",
        "htf_sweep_long",
        "htf_sweep_short",
        "htf_in_bull_fvg",
        "htf_in_bear_fvg",
        "htf_fvg_width",
        "htf_age_long",
        "htf_age_short",
        "htf_long_bias",
        "htf_short_bias",
        "session_ok",
        "vol_regime_ok",
        "long_setup",
        "short_setup",
        "long_ret",
        "short_ret",
        "long_hold_bars",
        "short_hold_bars",
        "long_risk_pct",
        "short_risk_pct",
    ]
    df = df[keep].replace([np.inf, -np.inf], np.nan)
    df = df[(df["htf_long_bias"] == 1) | (df["htf_short_bias"] == 1)].reset_index(drop=True)
    return df


def _max_dd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(float(len(returns))))


def _audit_one(cfg_raw: dict) -> dict:
    pair = cfg_raw["pair"]
    cfg = ResearchConfig(
        pair=pair,
        start="2024-01-01T00:00:00",
        end="2024-07-01T00:00:00",
        out_dir=f"models/production_portfolio/{cfg_raw['name']}",
        chunk_days=2,
        fetch_retries=3,
        fetch_timeout_sec=120.0,
        train_ratio=0.6,
        val_ratio=0.2,
        random_state=42,
        **{k: v for k, v in cfg_raw.items() if k not in {"pair", "name"}},
    )
    bars = pd.read_csv(f"models/compare_halfyear/cache/{pair.lower()}_bars_5m.csv", parse_dates=["ts"])
    bar_report = validate_ohlcv(bars)
    df = _build_audit_dataset(bars.copy(), cfg)

    mdir = Path(cfg.out_dir)
    meta = json.loads((mdir / "meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    threshold = float(meta["best_prob_threshold"])
    test_ts_start = pd.to_datetime(meta["test_ts_start"], utc=True)
    test = df[df["ts"] >= test_ts_start].copy().reset_index(drop=True)
    scaler = joblib.load(mdir / "scaler.joblib")
    model = joblib.load(mdir / "model.joblib")
    X = test[feat_cols].to_numpy(dtype=np.float64)
    proba = model.predict_proba(scaler.transform(X))

    fee = float(cfg.fee_bps) / 10000.0
    raw_rets: list[float] = []
    net_rets: list[float] = []
    risk1pct_rets: list[float] = []
    next_free = 0
    selected = 0
    risk_invalid = 0
    for i in range(len(test)):
        if i < next_free:
            continue
        p = proba[i]
        p_long = float(p[1]) if p.shape[0] > 1 else 0.0
        p_short = float(p[2]) if p.shape[0] > 2 else 0.0
        if max(p_long, p_short) < threshold:
            continue
        side = "long" if p_long >= p_short else "short"
        if side == "long":
            if not bool(test["long_setup"].iloc[i]):
                continue
            ret = float(test["long_ret"].iloc[i])
            risk_pct = float(test["long_risk_pct"].iloc[i]) if np.isfinite(test["long_risk_pct"].iloc[i]) else np.nan
            hold = int(test["long_hold_bars"].iloc[i])
        else:
            if not bool(test["short_setup"].iloc[i]):
                continue
            ret = float(test["short_ret"].iloc[i])
            risk_pct = float(test["short_risk_pct"].iloc[i]) if np.isfinite(test["short_risk_pct"].iloc[i]) else np.nan
            hold = int(test["short_hold_bars"].iloc[i])
        net = ret - fee
        raw_rets.append(ret)
        net_rets.append(net)
        if np.isfinite(risk_pct) and risk_pct > 1e-12:
            risk1pct_rets.append(0.01 * (net / risk_pct))
        else:
            risk_invalid += 1
        selected += 1
        next_free = i + max(hold, 1)

    def _summ(returns: list[float]) -> tuple[float, float, float]:
        if not returns:
            return 0.0, 0.0, 0.0
        arr = np.array(returns, dtype=np.float64)
        eq = np.cumprod(1.0 + arr)
        return float(eq[-1] - 1.0), _sharpe(arr), _max_dd(eq)

    raw_total, raw_sharpe, raw_mdd = _summ(raw_rets)
    net_total, net_sharpe, net_mdd = _summ(net_rets)
    risk_total, risk_sharpe, risk_mdd = _summ(risk1pct_rets)

    return {
        "pair": pair,
        "config_name": cfg_raw["name"],
        "bars_ok": bar_report.ok,
        "bars_rows": bar_report.n_rows,
        "bars_issues": "; ".join(bar_report.issues),
        "selected_trades": selected,
        "reported_test_total_return": meta["test_total_return"],
        "reported_test_sharpe": meta["test_sharpe"],
        "reported_test_max_drawdown": meta["test_max_drawdown"],
        "corrected_net_test_total_return": net_total,
        "corrected_net_test_sharpe": net_sharpe,
        "corrected_net_test_max_drawdown": net_mdd,
        "risk_1pct_total_return": risk_total,
        "risk_1pct_sharpe": risk_sharpe,
        "risk_1pct_max_drawdown": risk_mdd,
        "risk_1pct_end_equity": 1.0 + risk_total,
        "risk_1pct_profit_on_10k": 10000.0 * risk_total,
        "invalid_risk_rows": risk_invalid,
    }


def main() -> None:
    out_dir = Path("models/production_portfolio/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_audit_one(cfg) for cfg in PRODUCTION_CONFIGS]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_summary.csv", index=False)
    (out_dir / "audit_summary.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
