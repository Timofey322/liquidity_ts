from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class WeeklyAdaptiveConfig:
    pair: str
    bars_path: str
    out_dir: str
    fee_bps: float = 1.0
    train_weeks: int = 2
    move_horizon_bars: int = 36
    pre_lookback_bars: int = 12
    top_events: int = 30
    max_signals_per_week: int = 8
    min_event_move: float = 0.0015
    min_sl_atr: float = 0.6
    max_sl_atr: float = 3.0
    min_tp_atr: float = 0.8
    max_tp_atr: float = 6.0


def _load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    req = {"ts", "open", "high", "low", "close"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    return df


def _atr(df: pd.DataFrame, span: int = 14) -> np.ndarray:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr).ewm(span=span, adjust=False).mean().to_numpy()


def _week_start_utc(ts: pd.Series) -> pd.Series:
    day = ts.dt.floor("D")
    return day - pd.to_timedelta(ts.dt.weekday, unit="D")


def _extract_events(train_df: pd.DataFrame, cfg: WeeklyAdaptiveConfig) -> pd.DataFrame:
    n = len(train_df)
    h = int(cfg.move_horizon_bars)
    lb = int(cfg.pre_lookback_bars)
    if n < (h + lb + 5):
        return pd.DataFrame()

    close = train_df["close"].to_numpy(dtype=np.float64)
    high = train_df["high"].to_numpy(dtype=np.float64)
    low = train_df["low"].to_numpy(dtype=np.float64)
    atr = train_df["atr"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(train_df["ts"], utc=True)

    rows: list[dict] = []
    for i in range(lb, n - h):
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        fut_hi = np.max(high[i + 1 : i + 1 + h])
        fut_lo = np.min(low[i + 1 : i + 1 + h])
        up_move = (fut_hi / entry) - 1.0
        dn_move = 1.0 - (fut_lo / entry)
        side = 1 if up_move >= dn_move else -1
        move = up_move if side == 1 else dn_move
        if move < float(cfg.min_event_move):
            continue

        prev_close = close[i - lb : i]
        prev_high = high[i - lb : i]
        prev_low = low[i - lb : i]
        pre_ret = (entry / prev_close[0]) - 1.0
        pre_range_rel = (np.max(prev_high) - np.min(prev_low)) / (entry + 1e-12)
        prev_hi = float(np.max(prev_high))
        prev_lo = float(np.min(prev_low))
        breakout_up = bool(entry > prev_hi)
        breakout_dn = bool(entry < prev_lo)

        if side == 1:
            mae = max(0.0, (entry - fut_lo) / (atr[i] + 1e-12))
            mfe = max(0.0, (fut_hi - entry) / (atr[i] + 1e-12))
            breakout_match = breakout_up
            signed_pre_ret = pre_ret
        else:
            mae = max(0.0, (fut_hi - entry) / (atr[i] + 1e-12))
            mfe = max(0.0, (entry - fut_lo) / (atr[i] + 1e-12))
            breakout_match = breakout_dn
            signed_pre_ret = -pre_ret

        rows.append(
            {
                "idx": i,
                "ts": ts.iloc[i],
                "side": side,
                "move": move,
                "hour": int(ts.iloc[i].hour),
                "dow": int(ts.iloc[i].weekday()),
                "signed_pre_ret": float(signed_pre_ret),
                "pre_range_rel": float(pre_range_rel),
                "atr_rel": float(atr[i] / (entry + 1e-12)),
                "mae_atr": float(mae),
                "mfe_atr": float(mfe),
                "breakout_match": int(breakout_match),
            }
        )
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows).sort_values("move", ascending=False).head(int(cfg.top_events)).reset_index(drop=True)
    return events


def _build_pattern(events: pd.DataFrame, cfg: WeeklyAdaptiveConfig) -> dict | None:
    if events.empty:
        return None

    side_score = float(np.sum(events["side"] * events["move"]))
    dominant_side = 1 if side_score >= 0 else -1
    aligned = events[events["side"] == dominant_side].copy()
    if aligned.empty:
        aligned = events.copy()

    hour_counts = aligned["hour"].value_counts().sort_values(ascending=False)
    dow_counts = aligned["dow"].value_counts().sort_values(ascending=False)
    top_hours = hour_counts.head(3).index.astype(int).tolist()
    top_dows = dow_counts.head(3).index.astype(int).tolist()

    q = aligned[["signed_pre_ret", "pre_range_rel", "atr_rel"]].quantile([0.2, 0.8])
    sl_atr = float(np.clip(np.quantile(aligned["mae_atr"], 0.75), cfg.min_sl_atr, cfg.max_sl_atr))
    tp_atr = float(np.clip(np.quantile(aligned["mfe_atr"], 0.5), cfg.min_tp_atr, cfg.max_tp_atr))
    breakout_ratio = float(np.mean(aligned["breakout_match"] > 0))

    return {
        "dominant_side": int(dominant_side),
        "top_hours": top_hours,
        "top_dows": top_dows,
        "signed_pre_ret_lo": float(q.loc[0.2, "signed_pre_ret"]),
        "signed_pre_ret_hi": float(q.loc[0.8, "signed_pre_ret"]),
        "pre_range_rel_lo": float(q.loc[0.2, "pre_range_rel"]),
        "pre_range_rel_hi": float(q.loc[0.8, "pre_range_rel"]),
        "atr_rel_lo": float(q.loc[0.2, "atr_rel"]),
        "atr_rel_hi": float(q.loc[0.8, "atr_rel"]),
        "require_breakout": bool(breakout_ratio >= 0.5),
        "sl_atr": float(sl_atr),
        "tp_atr": float(tp_atr),
        "rr": float(tp_atr / (sl_atr + 1e-12)),
        "n_events": int(len(aligned)),
    }


def _simulate_trade(week_df: pd.DataFrame, idx: int, side: int, sl_atr: float, tp_atr: float, max_hold: int) -> tuple[int, float, float, str]:
    close = week_df["close"].to_numpy(dtype=np.float64)
    high = week_df["high"].to_numpy(dtype=np.float64)
    low = week_df["low"].to_numpy(dtype=np.float64)
    atr = week_df["atr"].to_numpy(dtype=np.float64)

    entry = float(close[idx])
    atr0 = float(atr[idx])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr0) or atr0 <= 0:
        return idx + 1, entry, entry, "invalid"

    if side > 0:
        stop = entry - sl_atr * atr0
        target = entry + tp_atr * atr0
    else:
        stop = entry + sl_atr * atr0
        target = entry - tp_atr * atr0

    last = min(idx + int(max_hold), len(week_df) - 1)
    exit_idx = last
    exit_px = float(close[last])
    reason = "time"
    for j in range(idx + 1, last + 1):
        if side > 0:
            if low[j] <= stop:
                exit_idx = j
                exit_px = stop
                reason = "stop"
                break
            if high[j] >= target:
                exit_idx = j
                exit_px = target
                reason = "target"
                break
        else:
            if high[j] >= stop:
                exit_idx = j
                exit_px = stop
                reason = "stop"
                break
            if low[j] <= target:
                exit_idx = j
                exit_px = target
                reason = "target"
                break
    return exit_idx, float(stop), float(exit_px), reason


def _apply_pattern(week_df: pd.DataFrame, pattern: dict, cfg: WeeklyAdaptiveConfig) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()

    close = week_df["close"].to_numpy(dtype=np.float64)
    high = week_df["high"].to_numpy(dtype=np.float64)
    low = week_df["low"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(week_df["ts"], utc=True)
    lb = int(cfg.pre_lookback_bars)
    h = int(cfg.move_horizon_bars)
    side = int(pattern["dominant_side"])

    trades: list[dict] = []
    next_free = lb
    signals = 0

    for i in range(lb, len(week_df) - 1):
        if i < next_free:
            continue
        if signals >= int(cfg.max_signals_per_week):
            break

        hour = int(ts.iloc[i].hour)
        dow = int(ts.iloc[i].weekday())
        if hour not in pattern["top_hours"] or dow not in pattern["top_dows"]:
            continue

        entry = float(close[i])
        prev_close = close[i - lb : i]
        prev_high = high[i - lb : i]
        prev_low = low[i - lb : i]
        signed_pre_ret = (entry / prev_close[0]) - 1.0
        if side < 0:
            signed_pre_ret = -signed_pre_ret
        pre_range_rel = (np.max(prev_high) - np.min(prev_low)) / (entry + 1e-12)
        atr_rel = float(week_df["atr"].iloc[i] / (entry + 1e-12))

        if not (pattern["signed_pre_ret_lo"] <= signed_pre_ret <= pattern["signed_pre_ret_hi"]):
            continue
        if not (pattern["pre_range_rel_lo"] <= pre_range_rel <= pattern["pre_range_rel_hi"]):
            continue
        if not (pattern["atr_rel_lo"] <= atr_rel <= pattern["atr_rel_hi"]):
            continue

        prev_hi = float(np.max(prev_high))
        prev_lo = float(np.min(prev_low))
        breakout_ok = True
        if bool(pattern["require_breakout"]):
            breakout_ok = entry > prev_hi if side > 0 else entry < prev_lo
        if not breakout_ok:
            continue

        exit_idx, stop, exit_px, reason = _simulate_trade(week_df, i, side, float(pattern["sl_atr"]), float(pattern["tp_atr"]), h)
        if not np.isfinite(exit_px) or exit_px <= 0:
            continue
        gross = (exit_px / entry) - 1.0 if side > 0 else 1.0 - (exit_px / entry)
        fee = float(cfg.fee_bps) / 10000.0
        net = gross - fee
        trades.append(
            {
                "entry_ts": ts.iloc[i],
                "exit_ts": ts.iloc[exit_idx],
                "side": "long" if side > 0 else "short",
                "entry": entry,
                "stop": stop,
                "exit_price": exit_px,
                "exit_reason": reason,
                "gross_ret": float(gross),
                "net_ret": float(net),
                "week_start": week_df["week_start"].iloc[0],
            }
        )
        next_free = exit_idx + 1
        signals += 1

    return pd.DataFrame(trades)


def _summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "sharpe": 0.0,
        }
    arr = trades["net_ret"].to_numpy(dtype=np.float64)
    eq = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    sharpe = 0.0
    if arr.size > 1:
        sd = float(np.std(arr, ddof=1))
        if sd > 1e-12:
            sharpe = float((float(np.mean(arr)) / sd) * np.sqrt(float(arr.size)))
    return {
        "n_trades": int(arr.size),
        "total_return": float(eq[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "win_rate": float(np.mean(arr > 0)),
        "avg_trade": float(np.mean(arr)),
        "sharpe": float(sharpe),
    }


def run_weekly_adaptive(cfg: WeeklyAdaptiveConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _load_bars(cfg.bars_path)
    df["atr"] = _atr(df)
    df["week_start"] = _week_start_utc(df["ts"])
    weeks = sorted(df["week_start"].dropna().unique().tolist())
    if len(weeks) <= int(cfg.train_weeks):
        raise ValueError("Not enough weeks for adaptive run.")

    all_trades: list[pd.DataFrame] = []
    pattern_rows: list[dict] = []

    for i in range(int(cfg.train_weeks), len(weeks)):
        train_weeks = weeks[i - int(cfg.train_weeks) : i]
        test_week = weeks[i]
        train_df = df[df["week_start"].isin(train_weeks)].reset_index(drop=True)
        test_df = df[df["week_start"] == test_week].reset_index(drop=True)
        if test_df.empty:
            continue

        events = _extract_events(train_df, cfg)
        pattern = _build_pattern(events, cfg)
        if pattern is None:
            pattern_rows.append({"week_start": str(test_week), "status": "skip_no_pattern"})
            continue

        trades_w = _apply_pattern(test_df, pattern, cfg)
        if not trades_w.empty:
            all_trades.append(trades_w)
        pattern_rows.append(
            {
                "week_start": str(test_week),
                "status": "ok",
                "dominant_side": int(pattern["dominant_side"]),
                "n_events": int(pattern["n_events"]),
                "rr": float(pattern["rr"]),
                "sl_atr": float(pattern["sl_atr"]),
                "tp_atr": float(pattern["tp_atr"]),
                "top_hours": pattern["top_hours"],
                "top_dows": pattern["top_dows"],
                "n_trades_week": int(len(trades_w)),
            }
        )

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = _summarize(trades)
    summary = {"pair": cfg.pair, "config": asdict(cfg)} | summary

    if not trades.empty:
        eq = pd.Series(
            np.cumprod(1.0 + trades["net_ret"].to_numpy(dtype=np.float64)),
            index=pd.to_datetime(trades["entry_ts"], utc=True),
        )
        eq.to_csv(out / "equity_curve.csv", index=True, header=["equity"])
        trades.to_csv(out / "trades.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq.index, eq.to_numpy(dtype=np.float64), color="black")
        ax.set_title(f"{cfg.pair} weekly adaptive liquidity equity")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "equity_curve.png", dpi=170)
        plt.close(fig)
    else:
        pd.Series([1.0], dtype=np.float64).to_csv(out / "equity_curve.csv", index=False, header=["equity"])

    pd.DataFrame(pattern_rows).to_csv(out / "weekly_patterns.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> WeeklyAdaptiveConfig:
    p = argparse.ArgumentParser(description="Weekly adaptive liquidity strategy")
    p.add_argument("--pair", default="XAUUSD")
    p.add_argument("--bars-path", default="models/compare_halfyear/cache/xauusd_bars_5m.csv")
    p.add_argument("--out-dir", default="models/weekly_adaptive/xauusd")
    p.add_argument("--fee-bps", type=float, default=2.0)
    p.add_argument("--train-weeks", type=int, default=2)
    p.add_argument("--move-horizon-bars", type=int, default=36)
    p.add_argument("--pre-lookback-bars", type=int, default=12)
    p.add_argument("--top-events", type=int, default=30)
    p.add_argument("--max-signals-per-week", type=int, default=8)
    p.add_argument("--min-event-move", type=float, default=0.0015)
    ns = p.parse_args()
    return WeeklyAdaptiveConfig(**vars(ns))


if __name__ == "__main__":
    res = run_weekly_adaptive(_parse_args())
    print(json.dumps(res, ensure_ascii=False, indent=2))
