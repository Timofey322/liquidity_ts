from __future__ import annotations

import numpy as np
import pandas as pd

from src.fast import footprint_histogram


def session_poc_from_ticks(ticks: pd.DataFrame, tick_size: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Маркет-профиль (упрощённо): POC = цена бина с максимальным суммарным объёмом по |signed|.
    """
    t = ticks.dropna(subset=["mid", "signed_size"]).copy()
    if t.empty:
        return float("nan"), np.array([]), np.array([])
    pmin = float(t["mid"].min())
    pmax = float(t["mid"].max())
    bid_v, ask_v, _, _ = footprint_histogram(
        t["mid"].to_numpy(dtype=np.float64),
        t["signed_size"].to_numpy(dtype=np.float64),
        float(tick_size),
        pmin,
        pmax,
    )
    total = bid_v + ask_v
    if total.sum() <= 0:
        return float("nan"), total, bid_v - ask_v
    idx = int(np.argmax(total))
    poc = pmin + idx * tick_size
    return poc, total, bid_v - ask_v
