from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class Tick:
    ts: pd.Timestamp
    bid: float
    ask: float
    bid_vol: float
    ask_vol: float


Side = Literal["buy", "sell", "unknown"]


@dataclass
class LevelCandidate:
    """Ценовой уровень: потенциал фиксации прибыли или убытка."""

    price: float
    kind: Literal["tp", "sl", "poc", "fractal", "imbalance_edge"]
    strength: float
    reason: str
