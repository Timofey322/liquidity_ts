from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.fast import detect_fvg_zones, fractal_highs, fractal_lows
from src.logic.levels import suggest_tp_sl
from src.logic.market_profile import session_poc_from_ticks
from src.logic.tick_flow import signed_flow_from_ticks


@dataclass
class BehaviorConfig:
    fractal_left: int = 2
    fractal_right: int = 2
    tick_size: float = 0.00005  # EURUSD ~ 0.5 pip; подстройте под инструмент
    gold_tick_size: float = 0.05  # XAUUSD грубо


def _pick_tick_size(pair: str, cfg: BehaviorConfig) -> float:
    p = pair.upper().replace("/", "")
    if p.startswith("XAU"):
        return float(cfg.gold_tick_size)
    return float(cfg.tick_size)


def _last_true_level(flag: np.ndarray, values: np.ndarray) -> Optional[float]:
    idx = np.where(flag)[0]
    if idx.size == 0:
        return None
    return float(values[idx[-1]])


def _price_in_zone(px: float, lo: float, hi: float) -> bool:
    return lo <= px <= hi


def analyze_session(
    ohlc: pd.DataFrame,
    ticks: pd.DataFrame,
    pair: str,
    cfg: BehaviorConfig | None = None,
) -> Dict[str, Any]:
    """
    Анализ взаимодействия:
    - пересечение уровня фрактала (по close на OHLC)
    - нахождение в зоне имбаланса (FVG)
    - футпринт: дельта signed_flow вокруг последнего бара
    - кандидаты TP/SL через suggest_tp_sl + POC
    """
    cfg = cfg or BehaviorConfig()
    tick_sz = _pick_tick_size(pair, cfg)

    df = ohlc.sort_values("ts").reset_index(drop=True)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    fh = fractal_highs(high, cfg.fractal_left, cfg.fractal_right)
    fl = fractal_lows(low, cfg.fractal_left, cfg.fractal_right)
    bull, bear, z_lo, z_hi = detect_fvg_zones(high, low, close)

    last = len(df) - 1
    last_close = float(close[last])

    last_fh = _last_true_level(fh[:last], high[:last]) if last > 0 else None
    last_fl = _last_true_level(fl[:last], low[:last]) if last > 0 else None

    # последняя сформированная зона FVG (ищем с конца)
    imb_lo, imb_hi = None, None
    for i in range(last, -1, -1):
        if bull[i] or bear[i]:
            lo = float(z_lo[i])
            hi = float(z_hi[i])
            if np.isfinite(lo) and np.isfinite(hi):
                imb_lo, imb_hi = lo, hi
                break

    # пересечение вверх последнего фрактала high (упрощение: текущий close > уровень и prev <=)
    cross_up_fractal = False
    cross_dn_fractal = False
    if last > 0:
        lim = last - cfg.fractal_right - 1
        for i in range(lim, -1, -1):
            if fh[i]:
                level = float(high[i])
                cross_up_fractal = bool(close[last] > level and close[last - 1] <= level)
                break
        for i in range(lim, -1, -1):
            if fl[i]:
                level = float(low[i])
                cross_dn_fractal = bool(close[last] < level and close[last - 1] >= level)
                break

    in_imbalance = False
    if imb_lo is not None and imb_hi is not None:
        in_imbalance = _price_in_zone(last_close, min(imb_lo, imb_hi), max(imb_lo, imb_hi))

    footprint_delta = 0.0
    poc = float("nan")
    if not ticks.empty:
        tdf = signed_flow_from_ticks(ticks)
        if not tdf.empty:
            poc, _, delta_bins = session_poc_from_ticks(tdf, tick_sz)
            footprint_delta = float(np.sum(delta_bins)) if delta_bins.size else 0.0
    else:
        o = df["open"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        if "volume" in df.columns:
            v = np.maximum(df["volume"].to_numpy(dtype=np.float64), 0.0)
        else:
            v = np.ones(len(df), dtype=np.float64)
        signed_bar = np.sign(c - o) * np.log1p(v)
        wnd = min(200, len(df))
        footprint_delta = float(np.sum(signed_bar[-wnd:]))
        w = v[-wnd:]
        cc = c[-wnd:]
        poc = float(np.sum(cc * w) / (np.sum(w) + 1e-12))

    direction = "flat"
    if cross_up_fractal and not cross_dn_fractal:
        direction = "long"
    elif cross_dn_fractal and not cross_up_fractal:
        direction = "short"

    levels = []
    if direction != "flat":
        levels = suggest_tp_sl(
            last_close,
            direction,
            last_fractal_high=last_fh,
            last_fractal_low=last_fl,
            imb_low=imb_lo,
            imb_high=imb_hi,
            poc=poc if np.isfinite(poc) else None,
        )

    return {
        "pair": pair,
        "last_ts": df["ts"].iloc[last],
        "last_close": last_close,
        "cross_up_fractal": cross_up_fractal,
        "cross_dn_fractal": cross_dn_fractal,
        "in_imbalance_zone": in_imbalance,
        "imbalance_low": imb_lo,
        "imbalance_high": imb_hi,
        "footprint_session_delta": footprint_delta,
        "poc": float(poc) if np.isfinite(poc) else None,
        "direction_hint": direction,
        "levels": levels,
    }
