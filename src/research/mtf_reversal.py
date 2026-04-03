from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.data.dukascopy import DukascopyConfig, fetch_ticks_range, ticks_to_dataframe
from src.data.resample import ticks_to_ohlcv
from src.fast import detect_fvg_zones, fractal_highs, fractal_lows
from src.logic.tick_flow import signed_flow_from_ticks

try:
    import optuna
except ImportError as e:  # pragma: no cover
    optuna = None
    _OPTUNA_ERR = e
else:
    _OPTUNA_ERR = None


@dataclass
class ResearchConfig:
    pair: str = "EURUSD"
    start: str = "2024-01-01T00:00:00"
    end: str = "2024-07-01T00:00:00"
    out_dir: str = "models/research_eurusd_mtf_halfyear"
    base_rule: str = "5min"
    htf_rule: str = "1h"
    chunk_days: int = 2
    fetch_retries: int = 3
    fetch_timeout_sec: float = 120.0
    fractal_left: int = 1
    fractal_right: int = 1
    idea_window_bars: int = 24
    max_hold_bars: int = 48
    fee_bps: float = 1.0
    use_sweep_idea: bool = True
    use_fvg_idea: bool = True
    footprint_threshold: float = 0.25
    entry_mode: str = "cross_or_ret"
    stop_atr_mult: float = 1.0
    target_rr: float = 1.5
    session_start_hour_utc: int | None = None
    session_end_hour_utc: int | None = None
    atr_quantile_window: int = 0
    min_atr_quantile: float = 0.0
    max_atr_quantile: float = 1.0
    n_trials: int = 20
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    min_prob_threshold: float = 0.45
    max_prob_threshold: float = 0.85
    footprint_z_window: int = 20
    random_state: int = 42


def _ts(value: str) -> pd.Timestamp:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("UTC")


def _confirmed_flags(flags: np.ndarray, values: np.ndarray, right: int) -> tuple[np.ndarray, np.ndarray]:
    delay = max(int(right) + 1, 0)
    out_flag = np.zeros(flags.shape[0], dtype=bool)
    out_val = np.full(flags.shape[0], np.nan, dtype=np.float64)
    if delay == 0:
        out_flag = flags.copy()
        out_val[flags] = values[flags]
        return out_flag, out_val
    if delay < flags.shape[0]:
        out_flag[delay:] = flags[:-delay]
        shifted = np.where(flags, values, np.nan)
        out_val[delay:] = shifted[:-delay]
    return out_flag, out_val


def _last_true_level(flag: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(flag.shape[0], np.nan, dtype=np.float64)
    last = np.nan
    for i in range(flag.shape[0]):
        if flag[i] and np.isfinite(values[i]):
            last = values[i]
        out[i] = last
    return out


def _in_last_fvg_zone(
    close: np.ndarray,
    bull: np.ndarray,
    bear: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    in_bull = np.zeros(n, dtype=bool)
    in_bear = np.zeros(n, dtype=bool)
    width = np.zeros(n, dtype=np.float64)
    last_lo = np.nan
    last_hi = np.nan
    last_is_bull = False
    for i in range(n):
        if bull[i] or bear[i]:
            lo = float(z_lo[i])
            hi = float(z_hi[i])
            if np.isfinite(lo) and np.isfinite(hi):
                last_lo, last_hi = lo, hi
                last_is_bull = bool(bull[i])
        if np.isfinite(last_lo) and np.isfinite(last_hi):
            lo, hi = min(last_lo, last_hi), max(last_lo, last_hi)
            inside = lo <= close[i] <= hi
            if last_is_bull:
                in_bull[i] = inside
            else:
                in_bear[i] = inside
            width[i] = hi - lo
    return in_bull, in_bear, width


def _cross_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fh: np.ndarray,
    fl: np.ndarray,
    fractal_right: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    cross_up = np.zeros(n, dtype=bool)
    cross_dn = np.zeros(n, dtype=bool)
    last_fh = np.nan
    last_fl = np.nan
    for i in range(1, n):
        c = i - fractal_right - 1
        if c >= 0:
            if fh[c]:
                last_fh = high[c]
            if fl[c]:
                last_fl = low[c]
        if np.isfinite(last_fh):
            cross_up[i] = bool(close[i] > last_fh and close[i - 1] <= last_fh)
        if np.isfinite(last_fl):
            cross_dn[i] = bool(close[i] < last_fl and close[i - 1] >= last_fl)
    return cross_up, cross_dn


def _resample_ohlcv(ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = ohlc.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    out = pd.DataFrame(
        {
            "open": df["open"].resample(rule).first(),
            "high": df["high"].resample(rule).max(),
            "low": df["low"].resample(rule).min(),
            "close": df["close"].resample(rule).last(),
            "volume": df["volume"].resample(rule).sum(),
            "footprint_raw": df["footprint_raw"].resample(rule).sum(),
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def _fetch_aggregated_bars(cfg: ResearchConfig) -> pd.DataFrame:
    start = _ts(cfg.start)
    end = _ts(cfg.end)
    cur = start
    parts: list[pd.DataFrame] = []
    duka_cfg = DukascopyConfig(timeout_sec=float(cfg.fetch_timeout_sec))
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=int(cfg.chunk_days)), end)
        last_err: Exception | None = None
        ticks = []
        for attempt in range(int(cfg.fetch_retries)):
            try:
                ticks = fetch_ticks_range(cfg.pair, cur.to_pydatetime(), nxt.to_pydatetime(), cfg=duka_cfg)
                last_err = None
                break
            except Exception as exc:  # pragma: no cover - network instability
                last_err = exc
                time.sleep(min(5 * (attempt + 1), 15))
        if last_err is not None:
            raise RuntimeError(
                f"Failed to fetch {cfg.pair} chunk {cur.isoformat()} -> {nxt.isoformat()}"
            ) from last_err
        df_ticks = ticks_to_dataframe(ticks)
        if not df_ticks.empty:
            ohlc = ticks_to_ohlcv(df_ticks, cfg.base_rule)
            flow = signed_flow_from_ticks(df_ticks)
            flow["bar_ts"] = pd.to_datetime(flow["ts"], utc=True).dt.floor(cfg.base_rule)
            fp = flow.groupby("bar_ts", sort=True)["signed_size"].sum().rename("footprint_raw").reset_index()
            fp = fp.rename(columns={"bar_ts": "ts"})
            chunk = ohlc.merge(fp, on="ts", how="left")
            chunk["footprint_raw"] = chunk["footprint_raw"].fillna(0.0)
            parts.append(chunk)
        print(f"[FETCH] {cfg.pair} {cur.isoformat()} -> {nxt.isoformat()} rows={0 if df_ticks.empty else len(df_ticks)}")
        cur = nxt
    if not parts:
        raise RuntimeError("No data fetched from Dukascopy.")
    bars = pd.concat(parts, ignore_index=True)
    bars = bars.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return bars


def _build_htf_context(base: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    htf = _resample_ohlcv(base, cfg.htf_rule)
    high = htf["high"].to_numpy(dtype=np.float64)
    low = htf["low"].to_numpy(dtype=np.float64)
    close = htf["close"].to_numpy(dtype=np.float64)
    volume = htf["volume"].to_numpy(dtype=np.float64)

    fh = fractal_highs(high, cfg.fractal_left, cfg.fractal_right)
    fl = fractal_lows(low, cfg.fractal_left, cfg.fractal_right)
    fh_flag, fh_val = _confirmed_flags(fh, high, cfg.fractal_right)
    fl_flag, fl_val = _confirmed_flags(fl, low, cfg.fractal_right)
    last_fh = _last_true_level(fh_flag, fh_val)
    last_fl = _last_true_level(fl_flag, fl_val)

    bull, bear, z_lo, z_hi = detect_fvg_zones(high, low, close)
    in_bull_fvg, in_bear_fvg, fvg_w = _in_last_fvg_zone(close, bull, bear, z_lo, z_hi)
    enter_bull_fvg = in_bull_fvg & ~np.r_[False, in_bull_fvg[:-1]]
    enter_bear_fvg = in_bear_fvg & ~np.r_[False, in_bear_fvg[:-1]]

    sweep_long = np.zeros(len(htf), dtype=bool)
    sweep_short = np.zeros(len(htf), dtype=bool)
    sweep_long_level = np.full(len(htf), np.nan, dtype=np.float64)
    sweep_short_level = np.full(len(htf), np.nan, dtype=np.float64)
    for i in range(len(htf)):
        prev_fh = last_fh[i - 1] if i > 0 else np.nan
        prev_fl = last_fl[i - 1] if i > 0 else np.nan
        if np.isfinite(prev_fl) and low[i] < prev_fl and close[i] > prev_fl:
            sweep_long[i] = True
            sweep_long_level[i] = low[i]
        if np.isfinite(prev_fh) and high[i] > prev_fh and close[i] < prev_fh:
            sweep_short[i] = True
            sweep_short_level[i] = high[i]

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()
    ret1 = np.zeros(len(htf), dtype=np.float64)
    ret1[1:] = (close[1:] / close[:-1]) - 1.0

    htf_df = pd.DataFrame(
        {
            "ts": pd.to_datetime(htf["ts"], utc=True),
            "htf_ret1": ret1,
            "htf_atr_rel": atr / (close + 1e-12),
            "htf_vol_log1p": np.log1p(np.maximum(volume, 0.0)),
            "htf_last_fh": last_fh,
            "htf_last_fl": last_fl,
            "htf_sweep_long": sweep_long.astype(np.int8),
            "htf_sweep_short": sweep_short.astype(np.int8),
            "htf_sweep_long_level": pd.Series(sweep_long_level).ffill().to_numpy(),
            "htf_sweep_short_level": pd.Series(sweep_short_level).ffill().to_numpy(),
            "htf_in_bull_fvg": in_bull_fvg.astype(np.int8),
            "htf_in_bear_fvg": in_bear_fvg.astype(np.int8),
            "htf_enter_bull_fvg": enter_bull_fvg.astype(np.int8),
            "htf_enter_bear_fvg": enter_bear_fvg.astype(np.int8),
            "htf_fvg_width": fvg_w / (close + 1e-12),
        }
    )
    long_idea = np.zeros(len(htf), dtype=bool)
    short_idea = np.zeros(len(htf), dtype=bool)
    if cfg.use_sweep_idea:
        long_idea |= sweep_long
        short_idea |= sweep_short
    if cfg.use_fvg_idea:
        long_idea |= enter_bull_fvg
        short_idea |= enter_bear_fvg
    htf_df["htf_last_long_idea_ts"] = pd.Series(np.where(long_idea, htf_df["ts"], pd.NaT)).ffill()
    htf_df["htf_last_short_idea_ts"] = pd.Series(np.where(short_idea, htf_df["ts"], pd.NaT)).ffill()
    return htf_df


def _simulate_side(
    df: pd.DataFrame,
    idx: int,
    side: str,
    cfg: ResearchConfig,
) -> tuple[float, int]:
    entry = float(df["close"].iloc[idx])
    atr = float(df["atr5"].iloc[idx])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return 0.0, 1

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
            return 0.0, 1
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
            return 0.0, 1
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
    return float(ret), int(max(hold, 1))


def _build_dataset(base: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
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
    for i in range(len(df) - 1):
        if bool(df["long_setup"].iloc[i]):
            df.loc[i, ["long_ret", "long_hold_bars"]] = _simulate_side(df, i, "long", cfg)
        if bool(df["short_setup"].iloc[i]):
            df.loc[i, ["short_ret", "short_hold_bars"]] = _simulate_side(df, i, "short", cfg)

    fee = float(cfg.fee_bps) / 10000.0 * 2.0
    net_long = df["long_ret"] - fee
    net_short = df["short_ret"] - fee
    y = np.zeros(len(df), dtype=np.int64)
    long_ok = (df["long_setup"]) & (net_long > 0) & (net_long >= net_short)
    short_ok = (df["short_setup"]) & (net_short > 0) & (net_short > net_long)
    y[long_ok.to_numpy()] = 1
    y[short_ok.to_numpy()] = 2
    df["y"] = y

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
        "y",
    ]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df = df[(df["htf_long_bias"] == 1) | (df["htf_short_bias"] == 1)].reset_index(drop=True)
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "ts",
        "open",
        "high",
        "low",
        "close",
        "long_ret",
        "short_ret",
        "long_hold_bars",
        "short_hold_bars",
        "y",
    }
    return [c for c in df.columns if c not in exclude]


def _time_splits(df: pd.DataFrame, cfg: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.sort_values("ts").reset_index(drop=True)
    purge = int(cfg.max_hold_bars)
    n_eff = len(d) - 2 * purge
    if n_eff <= 0:
        raise ValueError("Not enough rows after purge.")
    n_tr = int(n_eff * cfg.train_ratio)
    n_va = int(n_eff * cfg.val_ratio)
    n_te = n_eff - n_tr - n_va
    if n_tr < 100 or n_va < 50 or n_te < 50:
        raise ValueError(f"Too few rows for research split: {len(d)}")
    tr = d.iloc[:n_tr].copy()
    va = d.iloc[n_tr + purge : n_tr + purge + n_va].copy()
    te_start = n_tr + purge + n_va + purge
    te = d.iloc[te_start : te_start + n_te].copy()
    return tr, va, te


def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12)
    return float(dd.min())


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(float(returns.size)))


def _score(total_return: float, mdd: float, n_trades: int) -> float:
    if n_trades <= 0:
        return -1e9
    trade_factor = min(float(n_trades), 10.0) / 10.0
    return float(trade_factor * total_return / (1.0 + abs(mdd)))


def _backtest_candidates(
    df: pd.DataFrame,
    proba: np.ndarray,
    *,
    prob_threshold: float,
    fee_bps: float = 0.0,
    allow_overlap: bool = False,
) -> dict:
    rets: list[float] = []
    eq_index: list[pd.Timestamp] = []
    next_free = 0
    fee = float(fee_bps) / 10000.0
    for i in range(len(df)):
        if (not allow_overlap) and i < next_free:
            continue
        p = proba[i]
        p_long = float(p[1]) if p.shape[0] > 1 else 0.0
        p_short = float(p[2]) if p.shape[0] > 2 else 0.0
        if max(p_long, p_short) < float(prob_threshold):
            continue
        side = "long" if p_long >= p_short else "short"
        ret = float(df["long_ret"].iloc[i] if side == "long" else df["short_ret"].iloc[i])
        hold = int(df["long_hold_bars"].iloc[i] if side == "long" else df["short_hold_bars"].iloc[i])
        if side == "long" and not bool(df["long_setup"].iloc[i]):
            continue
        if side == "short" and not bool(df["short_setup"].iloc[i]):
            continue
        rets.append(ret - fee)
        eq_index.append(pd.to_datetime(df["ts"].iloc[i], utc=True))
        if not allow_overlap:
            next_free = i + max(hold, 1)
    if not rets:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "equity_curve": pd.Series([1.0], dtype=np.float64),
        }
    arr = np.array(rets, dtype=np.float64)
    equity = np.cumprod(1.0 + arr)
    return {
        "total_return": float(equity[-1] - 1.0),
        "sharpe": _sharpe(arr),
        "max_drawdown": _max_dd(equity),
        "n_trades": int(len(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "avg_trade": float(np.mean(arr)),
        "equity_curve": pd.Series(equity, index=eq_index),
    }


def _train_model(df: pd.DataFrame, cfg: ResearchConfig, out_dir: Path) -> dict:
    if optuna is None:
        raise RuntimeError("optuna is required") from _OPTUNA_ERR
    tr, va, te = _time_splits(df, cfg)
    feat_cols = _feature_cols(df)
    X_tr = tr[feat_cols].to_numpy(dtype=np.float64)
    y_tr = tr["y"].to_numpy(dtype=np.int64)
    X_va = va[feat_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "random_state": cfg.random_state,
        }
        clf = HistGradientBoostingClassifier(class_weight="balanced", **params)
        clf.fit(X_trs, y_tr)
        proba = clf.predict_proba(X_vas)
        prob_threshold = trial.suggest_float("prob_threshold", cfg.min_prob_threshold, cfg.max_prob_threshold)
        bt = _backtest_candidates(va, proba, prob_threshold=prob_threshold, fee_bps=float(cfg.fee_bps))
        score = _score(bt["total_return"], bt["max_drawdown"], bt["n_trades"])
        trial.set_user_attr("val_total_return", bt["total_return"])
        trial.set_user_attr("val_sharpe", bt["sharpe"])
        trial.set_user_attr("val_max_drawdown", bt["max_drawdown"])
        trial.set_user_attr("val_n_trades", bt["n_trades"])
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg.random_state)),
    )
    study.optimize(objective, n_trials=int(cfg.n_trials), show_progress_bar=False)

    best = study.best_params
    best_prob_threshold = float(best.pop("prob_threshold"))

    X_fit_raw = np.vstack([tr[feat_cols].to_numpy(dtype=np.float64), va[feat_cols].to_numpy(dtype=np.float64)])
    y_fit = np.concatenate([tr["y"].to_numpy(dtype=np.int64), va["y"].to_numpy(dtype=np.int64)])
    fit_scaler = StandardScaler()
    X_fit = fit_scaler.fit_transform(X_fit_raw)
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=cfg.random_state, **best)
    clf.fit(X_fit, y_fit)

    X_te = fit_scaler.transform(te[feat_cols].to_numpy(dtype=np.float64))
    te_proba = clf.predict_proba(X_te)
    test_bt = _backtest_candidates(te, te_proba, prob_threshold=best_prob_threshold, fee_bps=float(cfg.fee_bps))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "model.joblib")
    joblib.dump(fit_scaler, out_dir / "scaler.joblib")
    joblib.dump(feat_cols, out_dir / "feature_cols.joblib")
    test_bt["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=True, header=["equity"])

    meta = {
        "config": asdict(cfg),
        "feature_cols": feat_cols,
        "best_params": best,
        "best_prob_threshold": best_prob_threshold,
        "best_val_score": float(study.best_value),
        "best_val_total_return": float(study.best_trial.user_attrs.get("val_total_return", 0.0)),
        "best_val_sharpe": float(study.best_trial.user_attrs.get("val_sharpe", 0.0)),
        "best_val_max_drawdown": float(study.best_trial.user_attrs.get("val_max_drawdown", 0.0)),
        "best_val_n_trades": int(study.best_trial.user_attrs.get("val_n_trades", 0)),
        "test_total_return": float(test_bt["total_return"]),
        "test_sharpe": float(test_bt["sharpe"]),
        "test_max_drawdown": float(test_bt["max_drawdown"]),
        "test_n_trades": int(test_bt["n_trades"]),
        "test_win_rate": float(test_bt["win_rate"]),
        "test_avg_trade": float(test_bt["avg_trade"]),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "class_counts": {str(k): int(v) for k, v in df["y"].value_counts().items()},
        "test_ts_start": str(te["ts"].iloc[0]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run(cfg: ResearchConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _fetch_aggregated_bars(cfg)
    base.to_csv(out_dir / "bars_5m.csv", index=False)
    dataset = _build_dataset(base, cfg)
    dataset.to_csv(out_dir / "dataset.csv", index=False)
    if dataset.empty:
        raise RuntimeError("Empty dataset after applying HTF/LTF strategy logic.")
    meta = _train_model(dataset, cfg, out_dir)
    return meta


def _parse_args() -> ResearchConfig:
    p = argparse.ArgumentParser(description="HTF liquidity sweep + LTF confirmation research pipeline")
    p.add_argument("--pair", default="EURUSD")
    p.add_argument("--start", default="2024-01-01T00:00:00")
    p.add_argument("--end", default="2024-07-01T00:00:00")
    p.add_argument("--out-dir", default="models/research_eurusd_mtf_halfyear")
    p.add_argument("--base-rule", default="5min")
    p.add_argument("--htf-rule", default="1h")
    p.add_argument("--chunk-days", type=int, default=2)
    p.add_argument("--fetch-retries", type=int, default=3)
    p.add_argument("--fetch-timeout-sec", type=float, default=120.0)
    p.add_argument("--fractal-left", type=int, default=1)
    p.add_argument("--fractal-right", type=int, default=1)
    p.add_argument("--idea-window-bars", type=int, default=24)
    p.add_argument("--max-hold-bars", type=int, default=48)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--use-sweep-idea", action="store_true", default=True)
    p.add_argument("--no-use-sweep-idea", dest="use_sweep_idea", action="store_false")
    p.add_argument("--use-fvg-idea", action="store_true", default=True)
    p.add_argument("--no-use-fvg-idea", dest="use_fvg_idea", action="store_false")
    p.add_argument("--footprint-threshold", type=float, default=0.25)
    p.add_argument("--entry-mode", choices=["cross_only", "cross_or_ret", "cross_and_ret"], default="cross_or_ret")
    p.add_argument("--stop-atr-mult", type=float, default=1.0)
    p.add_argument("--target-rr", type=float, default=1.5)
    p.add_argument("--session-start-hour-utc", type=int, default=None)
    p.add_argument("--session-end-hour-utc", type=int, default=None)
    p.add_argument("--atr-quantile-window", type=int, default=0)
    p.add_argument("--min-atr-quantile", type=float, default=0.0)
    p.add_argument("--max-atr-quantile", type=float, default=1.0)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--min-prob-threshold", type=float, default=0.45)
    p.add_argument("--max-prob-threshold", type=float, default=0.85)
    p.add_argument("--footprint-z-window", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    ns = p.parse_args()
    return ResearchConfig(**vars(ns))


if __name__ == "__main__":
    result = run(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
