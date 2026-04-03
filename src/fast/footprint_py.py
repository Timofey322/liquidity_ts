from __future__ import annotations

import numpy as np


def footprint_histogram(
    price: np.ndarray,
    signed_size: np.ndarray,
    tick_size: float,
    price_min: float,
    price_max: float,
):
    nb = int((price_max - price_min) / tick_size) + 3
    nb = max(nb, 4)
    bid_v = np.zeros(nb, dtype=np.float64)
    ask_v = np.zeros(nb, dtype=np.float64)
    inv = 1.0 / tick_size
    for k in range(price.shape[0]):
        p = price[k]
        s = signed_size[k]
        if p < price_min or p > price_max:
            continue
        idx = int((p - price_min) * inv)
        idx = max(0, min(nb - 1, idx))
        if s >= 0:
            ask_v[idx] += s
        else:
            bid_v[idx] += -s
    return bid_v, ask_v, price_min, tick_size
