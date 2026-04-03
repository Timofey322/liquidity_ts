from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fast import detect_fvg_zones, fractal_highs, fractal_lows


@dataclass
class SessionPoiConfig:
    pair: str
    bars_path: str
    out_dir: str
    fee_bps: float
    poi_window_bars: int = 24
    body_threshold: float = 0.6
    stop_buffer_atr: float = 0.15
    target_rr: float = 2.0
    max_hold_bars: int = 72
    session_fractal_left: int = 1
    session_fractal_right: int = 1
    ltf_fractal_left: int = 1
    ltf_fractal_right: int = 1
    allowed_sessions: tuple[str, ...] = ("asia", "london", "newyork")
    use_fractal_sweep_poi: bool = True
    use_fvg_poi: bool = True
    use_bos_entry: bool = True
    use_choch_entry: bool = True
    use_inversion_entry: bool = True
    require_trend_filter: bool = True
    trend_lookback_bars: int = 12
    ema_span: int = 50


def _load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def _session_name(ts: pd.Timestamp) -> str:
    h = int(ts.hour)
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 13:
        return "london"
    if 13 <= h < 21:
        return "newyork"
    return "off"


def _build_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["session"] = tmp["ts"].map(_session_name)
    tmp = tmp[tmp["session"] != "off"].copy()
    tmp["session_date"] = tmp["ts"].dt.floor("D")
    tmp["session_id"] = tmp["session_date"].dt.strftime("%Y-%m-%d") + "_" + tmp["session"]
    return (
        tmp.groupby("session_id", sort=True)
        .agg(
            ts=("ts", "min"),
            session=("session", "first"),
            session_date=("session_date", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .reset_index(drop=True)
    )


def _bool_feature(df: pd.DataFrame, col: str, enabled: bool) -> pd.Series:
    if not enabled:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].astype(bool)


def _confirmed_flags(flags: np.ndarray, values: np.ndarray, right: int) -> tuple[np.ndarray, np.ndarray]:
    delay = max(int(right) + 1, 0)
    out_flag = np.zeros(flags.shape[0], dtype=bool)
    out_val = np.full(flags.shape[0], np.nan, dtype=np.float64)
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


def _daily_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.set_index("ts")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    daily["prev_day_high"] = daily["high"].shift(1)
    daily["prev_day_low"] = daily["low"].shift(1)
    return daily[["ts", "prev_day_high", "prev_day_low"]]


def build_session_poi_dataset(cfg: SessionPoiConfig) -> pd.DataFrame:
    df = _load_bars(cfg.bars_path)
    tr = np.maximum(
        df["high"].to_numpy(dtype=np.float64) - df["low"].to_numpy(dtype=np.float64),
        np.maximum(
            np.abs(df["high"].to_numpy(dtype=np.float64) - np.roll(df["close"].to_numpy(dtype=np.float64), 1)),
            np.abs(df["low"].to_numpy(dtype=np.float64) - np.roll(df["close"].to_numpy(dtype=np.float64), 1)),
        ),
    )
    tr[0] = float(df["high"].iloc[0] - df["low"].iloc[0])
    df["atr"] = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()
    df["ret1"] = pd.Series(df["close"]).pct_change(fill_method=None).fillna(0.0)
    df["ret6_sum"] = df["ret1"].rolling(6, min_periods=1).sum()
    df["body_rel"] = (df["close"] - df["open"]) / (df["atr"] + 1e-12)
    df["session"] = df["ts"].map(_session_name)
    allowed = set(cfg.allowed_sessions)
    df["is_session_bar"] = df["session"].isin(allowed).astype(np.int8)

    session_df = _build_session_frame(df)
    sh = session_df["high"].to_numpy(dtype=np.float64)
    sl = session_df["low"].to_numpy(dtype=np.float64)
    sc = session_df["close"].to_numpy(dtype=np.float64)
    sfh = fractal_highs(sh, cfg.session_fractal_left, cfg.session_fractal_right)
    sfl = fractal_lows(sl, cfg.session_fractal_left, cfg.session_fractal_right)
    sfh_flag, sfh_val = _confirmed_flags(sfh, sh, cfg.session_fractal_right)
    sfl_flag, sfl_val = _confirmed_flags(sfl, sl, cfg.session_fractal_right)
    session_df["sess_last_fractal_high"] = _last_true_level(sfh_flag, sfh_val)
    session_df["sess_last_fractal_low"] = _last_true_level(sfl_flag, sfl_val)
    bull, bear, z_lo, z_hi = detect_fvg_zones(sh, sl, sc)
    session_df["bull_fvg_lo_ffill"] = pd.Series(np.where(bull, np.minimum(z_lo, z_hi), np.nan)).ffill()
    session_df["bull_fvg_hi_ffill"] = pd.Series(np.where(bull, np.maximum(z_lo, z_hi), np.nan)).ffill()
    session_df["bear_fvg_lo_ffill"] = pd.Series(np.where(bear, np.minimum(z_lo, z_hi), np.nan)).ffill()
    session_df["bear_fvg_hi_ffill"] = pd.Series(np.where(bear, np.maximum(z_lo, z_hi), np.nan)).ffill()

    df = pd.merge_asof(
        df.sort_values("ts"),
        session_df[
            [
                "ts",
                "sess_last_fractal_high",
                "sess_last_fractal_low",
                "bull_fvg_lo_ffill",
                "bull_fvg_hi_ffill",
                "bear_fvg_lo_ffill",
                "bear_fvg_hi_ffill",
            ]
        ].sort_values("ts"),
        on="ts",
        direction="backward",
    )
    df = pd.merge_asof(df, _daily_liquidity(df).sort_values("ts"), on="ts", direction="backward")

    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    fh = fractal_highs(h, cfg.ltf_fractal_left, cfg.ltf_fractal_right)
    fl = fractal_lows(l, cfg.ltf_fractal_left, cfg.ltf_fractal_right)
    fh_flag, fh_val = _confirmed_flags(fh, h, cfg.ltf_fractal_right)
    fl_flag, fl_val = _confirmed_flags(fl, l, cfg.ltf_fractal_right)
    df["last_ltf_fh"] = _last_true_level(fh_flag, fh_val)
    df["last_ltf_fl"] = _last_true_level(fl_flag, fl_val)
    df["bos_up"] = (df["close"] > df["last_ltf_fh"]) & (df["close"].shift(1) <= df["last_ltf_fh"])
    df["bos_dn"] = (df["close"] < df["last_ltf_fl"]) & (df["close"].shift(1) >= df["last_ltf_fl"])
    df["choch_up"] = df["bos_up"] & (df["ret6_sum"] < 0)
    df["choch_dn"] = df["bos_dn"] & (df["ret6_sum"] > 0)
    df["inv_long"] = (df["body_rel"] > cfg.body_threshold) & (df["ret1"].shift(1) < 0)
    df["inv_short"] = (df["body_rel"] < -cfg.body_threshold) & (df["ret1"].shift(1) > 0)

    # Simple trend continuation filter (LTF): require recent net drift in trade direction.
    trend_lb = max(int(cfg.trend_lookback_bars), 1)
    df["trend_up"] = df["ret1"].rolling(trend_lb, min_periods=trend_lb).sum() > 0
    df["trend_dn"] = df["ret1"].rolling(trend_lb, min_periods=trend_lb).sum() < 0
    ema_span = max(int(cfg.ema_span), 2)
    df["ema"] = pd.Series(df["close"].to_numpy(dtype=np.float64)).ewm(span=ema_span, adjust=False).mean().to_numpy()
    df["ema_up"] = df["close"] > df["ema"]
    df["ema_dn"] = df["close"] < df["ema"]

    bull_fvg_inside = (
        np.isfinite(df["bull_fvg_lo_ffill"])
        & np.isfinite(df["bull_fvg_hi_ffill"])
        & (df["low"] <= df["bull_fvg_hi_ffill"])
        & (df["high"] >= df["bull_fvg_lo_ffill"])
    )
    bear_fvg_inside = (
        np.isfinite(df["bear_fvg_lo_ffill"])
        & np.isfinite(df["bear_fvg_hi_ffill"])
        & (df["high"] >= df["bear_fvg_lo_ffill"])
        & (df["low"] <= df["bear_fvg_hi_ffill"])
    )
    # Continuation flow (liquidity->liquidity):
    # 1) trend continuation context
    # 2) FVG (imbalance) retest
    # 3) sweep nearest opposite-side LTF fractal (liquidity grab)
    # 4) BOS in trend direction
    #
    # Long continuation: bull FVG touch -> sweep LTF low fractal -> BOS up.
    # Short continuation: bear FVG touch -> sweep LTF high fractal -> BOS down.

    df["fvg_touch_long"] = bull_fvg_inside & (df["is_session_bar"] == 1)
    df["fvg_touch_short"] = bear_fvg_inside & (df["is_session_bar"] == 1)

    df["sweep_low_ltf"] = (df["low"] < df["last_ltf_fl"]) & (df["close"] > df["last_ltf_fl"]) & (df["is_session_bar"] == 1)
    df["sweep_high_ltf"] = (df["high"] > df["last_ltf_fh"]) & (df["close"] < df["last_ltf_fh"]) & (df["is_session_bar"] == 1)

    # Track last index of each step to enforce ordering without look-ahead.
    idx = np.arange(len(df), dtype=np.int64)
    last_fvg_long = pd.Series(np.where(df["fvg_touch_long"], idx, np.nan)).ffill().to_numpy(dtype=np.float64)
    last_fvg_short = pd.Series(np.where(df["fvg_touch_short"], idx, np.nan)).ffill().to_numpy(dtype=np.float64)
    last_sweep_low = pd.Series(np.where(df["sweep_low_ltf"], idx, np.nan)).ffill().to_numpy(dtype=np.float64)
    last_sweep_high = pd.Series(np.where(df["sweep_high_ltf"], idx, np.nan)).ffill().to_numpy(dtype=np.float64)

    window = float(max(int(cfg.poi_window_bars), 1))
    bos_up = _bool_feature(df, "bos_up", cfg.use_bos_entry)
    bos_dn = _bool_feature(df, "bos_dn", cfg.use_bos_entry)

    long_seq_ok = (
        np.isfinite(last_fvg_long)
        & np.isfinite(last_sweep_low)
        & (last_sweep_low >= last_fvg_long)
        & ((last_sweep_low - last_fvg_long) <= window)
        & ((idx - last_sweep_low) <= window)
    )
    short_seq_ok = (
        np.isfinite(last_fvg_short)
        & np.isfinite(last_sweep_high)
        & (last_sweep_high >= last_fvg_short)
        & ((last_sweep_high - last_fvg_short) <= window)
        & ((idx - last_sweep_high) <= window)
    )

    trend_ok_long = (~cfg.require_trend_filter) | (df["trend_up"].fillna(False) & df["ema_up"].fillna(False))
    trend_ok_short = (~cfg.require_trend_filter) | (df["trend_dn"].fillna(False) & df["ema_dn"].fillna(False))

    df["long_entry"] = trend_ok_long & long_seq_ok & bos_up
    df["short_entry"] = trend_ok_short & short_seq_ok & bos_dn
    return df


def _next_target_long(row: pd.Series) -> float | None:
    candidates = []
    if np.isfinite(row.get("sess_last_fractal_high", np.nan)) and row["sess_last_fractal_high"] > row["close"]:
        candidates.append(float(row["sess_last_fractal_high"]))
    if np.isfinite(row.get("prev_day_high", np.nan)) and row["prev_day_high"] > row["close"]:
        candidates.append(float(row["prev_day_high"]))
    return min(candidates) if candidates else None


def _next_target_short(row: pd.Series) -> float | None:
    candidates = []
    if np.isfinite(row.get("sess_last_fractal_low", np.nan)) and row["sess_last_fractal_low"] < row["close"]:
        candidates.append(float(row["sess_last_fractal_low"]))
    if np.isfinite(row.get("prev_day_low", np.nan)) and row["prev_day_low"] < row["close"]:
        candidates.append(float(row["prev_day_low"]))
    return max(candidates) if candidates else None


def simulate_session_poi_trades(df: pd.DataFrame, cfg: SessionPoiConfig) -> tuple[pd.DataFrame, pd.Series]:
    trades: list[dict] = []
    next_free = 0
    fee = float(cfg.fee_bps) / 10000.0
    n = len(df)
    for i in range(n - 1):
        if i < next_free:
            continue
        if bool(df["long_entry"].iloc[i]):
            side = "long"
            entry = float(df["close"].iloc[i])
            fractal = float(df["last_ltf_fl"].iloc[i]) if np.isfinite(df["last_ltf_fl"].iloc[i]) else np.nan
            if not np.isfinite(fractal) or fractal >= entry:
                continue
            stop = fractal - float(cfg.stop_buffer_atr) * float(df["atr"].iloc[i])
            risk = entry - stop
            if risk <= 1e-12:
                continue
            target = entry + float(cfg.target_rr) * risk
        elif bool(df["short_entry"].iloc[i]):
            side = "short"
            entry = float(df["close"].iloc[i])
            fractal = float(df["last_ltf_fh"].iloc[i]) if np.isfinite(df["last_ltf_fh"].iloc[i]) else np.nan
            if not np.isfinite(fractal) or fractal <= entry:
                continue
            stop = fractal + float(cfg.stop_buffer_atr) * float(df["atr"].iloc[i])
            risk = stop - entry
            if risk <= 1e-12:
                continue
            target = entry - float(cfg.target_rr) * risk
        else:
            continue

        exit_idx = min(i + int(cfg.max_hold_bars), n - 1)
        exit_price = float(df["close"].ffill().iloc[exit_idx])
        exit_reason = "time"
        for j in range(i + 1, exit_idx + 1):
            lo = float(df["low"].iloc[j])
            hi = float(df["high"].iloc[j])
            if side == "long":
                if lo <= stop:
                    exit_idx = j
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if hi >= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target"
                    break
            else:
                if hi >= stop:
                    exit_idx = j
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if lo <= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target"
                    break
        if not np.isfinite(exit_price) or exit_price <= 0:
            continue
        gross_ret = (exit_price / entry) - 1.0 if side == "long" else 1.0 - (exit_price / entry)
        net_ret = gross_ret - fee
        risk_pct = risk / entry
        trades.append(
            {
                "entry_idx": i,
                "exit_idx": exit_idx,
                "entry_ts": df["ts"].iloc[i],
                "exit_ts": df["ts"].iloc[exit_idx],
                "side": side,
                "entry": entry,
                "stop": stop,
                "target": target,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "risk_pct": risk_pct,
                "r_multiple": net_ret / (risk_pct + 1e-12),
            }
        )
        next_free = exit_idx + 1
    if not trades:
        return pd.DataFrame(), pd.Series([1.0], dtype=np.float64)
    tdf = pd.DataFrame(trades)
    eq = pd.Series(np.cumprod(1.0 + tdf["net_ret"].to_numpy(dtype=np.float64)), index=pd.to_datetime(tdf["entry_ts"], utc=True))
    return tdf, eq


def summarize_trades(tdf: pd.DataFrame, eq: pd.Series) -> dict:
    if tdf.empty:
        return {"n_trades": 0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade": 0.0}
    arr = tdf["net_ret"].to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(eq.to_numpy(dtype=np.float64))
    dd = (eq.to_numpy(dtype=np.float64) - peak) / (peak + 1e-12)
    sharpe = 0.0
    if len(arr) > 1:
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        if sd > 1e-12:
            sharpe = float((mu / sd) * np.sqrt(float(len(arr))))
    return {
        "n_trades": int(len(tdf)),
        "total_return": float(eq.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "win_rate": float(np.mean(tdf["net_ret"] > 0)),
        "avg_trade": float(np.mean(arr)),
        "avg_r_multiple": float(np.mean(tdf["r_multiple"])),
    }


def plot_trade_examples(df: pd.DataFrame, trades: pd.DataFrame, out_dir: str | Path, pair: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if trades.empty:
        return
    winners = trades[trades["net_ret"] > 0]
    losers = trades[trades["net_ret"] <= 0]
    rng = random.Random(42)
    samples = []
    if not winners.empty:
        samples.append(("winning_trade", winners.iloc[rng.randrange(len(winners))]))
    if not losers.empty:
        samples.append(("losing_trade", losers.iloc[rng.randrange(len(losers))]))
    for label, trade in samples:
        i0 = max(int(trade["entry_idx"]) - 20, 0)
        i1 = min(int(trade["exit_idx"]) + 20, len(df) - 1)
        window = df.iloc[i0 : i1 + 1].copy()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(window["ts"], window["close"], color="black", linewidth=1.2, label="close")
        ax.axhline(float(trade["entry"]), color="blue", linestyle="--", label="entry")
        ax.axhline(float(trade["stop"]), color="red", linestyle="--", label="stop")
        ax.axhline(float(trade["target"]), color="green", linestyle="--", label="take")
        ax.axvline(pd.to_datetime(trade["entry_ts"], utc=True), color="blue", alpha=0.4)
        ax.axvline(pd.to_datetime(trade["exit_ts"], utc=True), color="purple", alpha=0.4)
        ax.set_title(f"{pair} {label} | side={trade['side']} | exit={trade['exit_reason']} | net={trade['net_ret']:.4f}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{pair.lower()}_{label}.png", dpi=160)
        plt.close(fig)


def run_session_poi(cfg: SessionPoiConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_session_poi_dataset(cfg)
    df.to_csv(out / "dataset.csv", index=False)
    trades, eq = simulate_session_poi_trades(df, cfg)
    trades.to_csv(out / "trades.csv", index=False)
    eq.to_csv(out / "equity_curve.csv", index=True, header=["equity"])
    plot_trade_examples(df, trades, out / "charts", cfg.pair)
    summary = {"pair": cfg.pair, "config": cfg.__dict__} | summarize_trades(trades, eq)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
