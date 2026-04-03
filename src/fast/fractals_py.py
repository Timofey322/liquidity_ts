from __future__ import annotations

import numpy as np


def fractal_highs(high: np.ndarray, left: int, right: int) -> np.ndarray:
    n = high.shape[0]
    out = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        hi = high[i]
        if np.any(hi <= high[i - left : i]):
            continue
        if np.any(hi <= high[i + 1 : i + right + 1]):
            continue
        out[i] = True
    return out


def fractal_lows(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = low.shape[0]
    out = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        lo = low[i]
        if np.any(lo >= low[i - left : i]):
            continue
        if np.any(lo >= low[i + 1 : i + right + 1]):
            continue
        out[i] = True
    return out
