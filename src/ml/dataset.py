from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.fast import detect_fvg_zones, fractal_highs, fractal_lows
from src.logic.behavior import BehaviorConfig
from src.logic.tick_flow import signed_flow_from_ticks


@dataclass
class DatasetConfig:
    fractal_left: int = 2
    fractal_right: int = 2
    horizon: int = 12  # баров вперёд для метки
    ret_threshold: float = 0.0003  # 0.03% — порог «не flat» для FX 1m
    footprint_window_bars: int = 5  # сумма дельты футпринта за N баров
    htf_rule: str = "1h"  # higher timeframe для мультитаймфрейм-контекста


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


def _in_last_fvg_zone(
    close: np.ndarray,
    bull: np.ndarray,
    bear: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """На каждом баре: находимся ли в последней известной зоне FVG + ширина."""
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
            px = close[i]
            inside = lo <= px <= hi
            if last_is_bull:
                in_bull[i] = inside
            else:
                in_bear[i] = inside
            width[i] = hi - lo
    return in_bull, in_bear, width


def _confirmed_fractal_flags(flags: np.ndarray, right: int) -> np.ndarray:
    """
    Фрактал в точке i становится подтвержденным только после right следующих баров.
    Добавляем ещё 1 бар задержки, чтобы фича на баре t совпадала с консервативной
    логикой `_cross_series`, где уровень начинает использоваться начиная с t+1.
    """
    delay = max(int(right) + 1, 0)
    out = np.zeros(flags.shape[0], dtype=bool)
    if delay == 0:
        return flags.copy()
    if delay < flags.shape[0]:
        out[delay:] = flags[:-delay]
    return out


def _footprint_proxy_from_ohlc(ohlc: pd.DataFrame, window: int) -> pd.Series:
    o = ohlc["open"].astype(np.float64)
    c = ohlc["close"].astype(np.float64)
    if "volume" in ohlc.columns:
        v = ohlc["volume"].astype(np.float64).clip(lower=0.0)
    else:
        v = pd.Series(np.ones(len(ohlc), dtype=np.float64), index=ohlc.index)
    signed = np.sign(c - o) * np.log1p(v)
    return signed.rolling(int(window), min_periods=1).sum()


def _infer_bar_rule(ohlc: pd.DataFrame) -> str:
    ts = pd.to_datetime(ohlc["ts"], utc=True, errors="coerce").dropna()
    if len(ts) < 3:
        return "1min"
    diffs = ts.diff().dropna()
    if diffs.empty:
        return "1min"
    step = diffs.mode().iloc[0]
    if step <= pd.Timedelta(0):
        return "1min"
    total_seconds = int(step.total_seconds())
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def _footprint_roll(ohlc: pd.DataFrame, ticks: pd.DataFrame | None, window: int) -> pd.Series:
    if ticks is None or ticks.empty:
        return _footprint_proxy_from_ohlc(ohlc, window)
    t = signed_flow_from_ticks(ticks)
    bar_rule = _infer_bar_rule(ohlc)
    t["bar_ts"] = pd.to_datetime(t["ts"], utc=True).dt.floor(bar_rule)
    g = t.groupby("bar_ts", sort=True)["signed_size"].sum()
    vals = []
    for ts in pd.to_datetime(ohlc["ts"], utc=True):
        key = ts.floor(bar_rule)
        vals.append(float(g.get(key, 0.0)))
    s = pd.Series(vals, index=ohlc.index, dtype=np.float64)
    return s.rolling(int(window), min_periods=1).sum()


def _resample_ohlcv(ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = ohlc.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df = df.set_index("ts")
    out = pd.DataFrame(
        {
            "open": df["open"].resample(rule).first(),
            "high": df["high"].resample(rule).max(),
            "low": df["low"].resample(rule).min(),
            "close": df["close"].resample(rule).last(),
            "volume": df["volume"].resample(rule).sum()
            if "volume" in df.columns
            else pd.Series(0.0, index=df["close"].resample(rule).last().index),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()


def _higher_tf_features(ohlc: pd.DataFrame, rule: str, beh: BehaviorConfig) -> pd.DataFrame:
    htf = _resample_ohlcv(ohlc, rule)
    if htf.empty:
        return pd.DataFrame(columns=["ts"])

    high = htf["high"].to_numpy(dtype=np.float64)
    low = htf["low"].to_numpy(dtype=np.float64)
    close = htf["close"].to_numpy(dtype=np.float64)
    volume = htf["volume"].to_numpy(dtype=np.float64) if "volume" in htf.columns else np.zeros(len(htf))

    fh = fractal_highs(high, beh.fractal_left, beh.fractal_right)
    fl = fractal_lows(low, beh.fractal_left, beh.fractal_right)
    fh_known = _confirmed_fractal_flags(fh, beh.fractal_right)
    fl_known = _confirmed_fractal_flags(fl, beh.fractal_right)
    bull, bear, z_lo, z_hi = detect_fvg_zones(high, low, close)
    cross_up, cross_dn = _cross_series(high, low, close, fh, fl, beh.fractal_right)
    in_bull_fvg, in_bear_fvg, fvg_w = _in_last_fvg_zone(close, bull, bear, z_lo, z_hi)

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()
    ret1 = np.zeros(len(htf), dtype=np.float64)
    ret1[1:] = (close[1:] / close[:-1]) - 1.0

    return pd.DataFrame(
        {
            "ts": pd.to_datetime(htf["ts"], utc=True),
            "htf_ret1": ret1,
            "htf_atr_rel": atr / (close + 1e-12),
            "htf_vol_log1p": np.log1p(np.maximum(volume, 0.0)),
            "htf_fh": fh_known.astype(np.int8),
            "htf_fl": fl_known.astype(np.int8),
            "htf_cross_up": cross_up.astype(np.int8),
            "htf_cross_dn": cross_dn.astype(np.int8),
            "htf_in_bull_fvg": in_bull_fvg.astype(np.int8),
            "htf_in_bear_fvg": in_bear_fvg.astype(np.int8),
            "htf_fvg_width": fvg_w / (close + 1e-12),
        }
    )


def build_liquidity_dataset(
    ohlc: pd.DataFrame,
    pair: str,
    ticks: pd.DataFrame | None = None,
    cfg: DatasetConfig | None = None,
    beh: BehaviorConfig | None = None,
) -> pd.DataFrame:
    """
    Таблица для ML: фичи на закрытии бара i, метка по движению к i+horizon.
    Классы: 0=flat, 1=up, 2=down (по порогу доходности).
    """
    cfg = cfg or DatasetConfig()
    beh = beh or BehaviorConfig()

    df = ohlc.sort_values("ts").reset_index(drop=True)
    n = len(df)
    if n < cfg.horizon + cfg.fractal_left + cfg.fractal_right + 5:
        return pd.DataFrame()

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else np.zeros(n)

    fh = fractal_highs(high, beh.fractal_left, beh.fractal_right)
    fl = fractal_lows(low, beh.fractal_left, beh.fractal_right)
    fh_known = _confirmed_fractal_flags(fh, beh.fractal_right)
    fl_known = _confirmed_fractal_flags(fl, beh.fractal_right)
    bull, bear, z_lo, z_hi = detect_fvg_zones(high, low, close)
    cross_up, cross_dn = _cross_series(high, low, close, fh, fl, beh.fractal_right)
    in_bull_fvg, in_bear_fvg, fvg_w = _in_last_fvg_zone(close, bull, bear, z_lo, z_hi)

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()

    ret1 = np.zeros(n)
    ret1[1:] = (close[1:] / close[:-1]) - 1.0
    ret3 = pd.Series(close).pct_change(3, fill_method=None).to_numpy()
    ret6 = pd.Series(close).pct_change(6, fill_method=None).to_numpy()

    fp_roll = _footprint_roll(df, ticks, cfg.footprint_window_bars).to_numpy()
    htf = _higher_tf_features(df, cfg.htf_rule, beh)

    fut = np.full(n, np.nan)
    h = int(cfg.horizon)
    fut[:-h] = (close[h:] / close[:-h]) - 1.0

    thr = float(cfg.ret_threshold)
    y = np.zeros(n, dtype=np.int64)
    y[np.isfinite(fut) & (fut > thr)] = 1
    y[np.isfinite(fut) & (fut < -thr)] = 2

    warmup = max(20, beh.fractal_left + beh.fractal_right + 5)
    idx_lo = warmup
    idx_hi = n - h
    close_exit = close[idx_lo + h : idx_hi + h]

    rows = pd.DataFrame(
        {
            "ts": df["ts"].iloc[idx_lo:idx_hi].to_numpy(),
            "close": close[idx_lo:idx_hi],
            "close_exit": close_exit,
            "ret1": ret1[idx_lo:idx_hi],
            "ret3": ret3[idx_lo:idx_hi],
            "ret6": ret6[idx_lo:idx_hi],
            "atr_rel": (atr / (close + 1e-12))[idx_lo:idx_hi],
            "vol_log1p": np.log1p(np.maximum(vol, 0.0))[idx_lo:idx_hi],
            "fh": fh_known[idx_lo:idx_hi].astype(np.int8),
            "fl": fl_known[idx_lo:idx_hi].astype(np.int8),
            "cross_up": cross_up[idx_lo:idx_hi].astype(np.int8),
            "cross_dn": cross_dn[idx_lo:idx_hi].astype(np.int8),
            "in_bull_fvg": in_bull_fvg[idx_lo:idx_hi].astype(np.int8),
            "in_bear_fvg": in_bear_fvg[idx_lo:idx_hi].astype(np.int8),
            "fvg_width": (fvg_w / (close + 1e-12))[idx_lo:idx_hi],
            "footprint_roll": fp_roll[idx_lo:idx_hi],
            "y": y[idx_lo:idx_hi],
            "fut_ret": fut[idx_lo:idx_hi],
        }
    )
    if not htf.empty:
        rows["ts"] = pd.to_datetime(rows["ts"], utc=True)
        htf["ts"] = pd.to_datetime(htf["ts"], utc=True)
        rows = pd.merge_asof(
            rows.sort_values("ts"),
            htf.sort_values("ts"),
            on="ts",
            direction="backward",
        )
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna()
    rows["pair"] = pair
    return rows.reset_index(drop=True)


def feature_columns(ds: pd.DataFrame) -> list[str]:
    return [
        c
        for c in ds.columns
        if c
        not in ("ts", "y", "fut_ret", "pair", "close", "close_exit", "bar_pos")
    ]
