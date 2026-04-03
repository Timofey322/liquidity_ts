from __future__ import annotations

import pandas as pd


def ticks_to_ohlcv(ticks: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """
    OHLCV из тиков: mid = (bid+ask)/2, объём ~ ask_vol+bid_vol (как прокси).
    """
    if ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    t = ticks.copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    t = t.sort_values("ts")
    t["mid"] = (t["bid"].astype(float) + t["ask"].astype(float)) / 2.0
    t["vol"] = t["ask_vol"].astype(float).abs() + t["bid_vol"].astype(float).abs()
    t = t.set_index("ts")
    o = t["mid"].resample(rule).first()
    h = t["mid"].resample(rule).max()
    l = t["mid"].resample(rule).min()
    c = t["mid"].resample(rule).last()
    v = t["vol"].resample(rule).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna(how="all")
    out = out.reset_index()
    return out
