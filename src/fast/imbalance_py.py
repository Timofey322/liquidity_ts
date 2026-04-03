from __future__ import annotations

import numpy as np


def detect_fvg_zones(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,  # noqa: ARG001 — единый API с Cython-версией
):
    n = high.shape[0]
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    z_lo = np.full(n, np.nan)
    z_hi = np.full(n, np.nan)
    for i in range(2, n):
        h2 = high[i - 2]
        l2 = low[i - 2]
        hi = high[i]
        li = low[i]
        if li > h2:
            bull[i] = True
            z_lo[i] = h2
            z_hi[i] = li
        elif hi < l2:
            bear[i] = True
            z_lo[i] = hi
            z_hi[i] = l2
    return bull, bear, z_lo, z_hi
