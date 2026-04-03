from __future__ import annotations

import numpy as np
import pandas as pd


def signed_flow_from_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Прокси агрессора по движению mid и объёму тика (не Level-2, но даёт след сделок для футпринта).
    """
    t = ticks.sort_values("ts").copy()
    mid = (t["bid"].astype(float) + t["ask"].astype(float)) / 2.0
    vol = t["ask_vol"].astype(float).abs() + t["bid_vol"].astype(float).abs()
    vol = vol.replace(0.0, 1e-12)
    dm = mid.diff()
    signed = np.where(dm > 0, vol, np.where(dm < 0, -vol, 0.0))
    t["mid"] = mid
    t["signed_size"] = signed
    return t
