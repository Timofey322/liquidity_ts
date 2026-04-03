from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class OhlcvValidationReport:
    ok: bool
    n_rows: int
    issues: List[str]


def validate_ohlcv(df: pd.DataFrame) -> OhlcvValidationReport:
    """
    Проверки: монотонность времени, OHLC логика, положительные цены, дубликаты ts.
    """
    issues: List[str] = []
    req = {"ts", "open", "high", "low", "close"}
    if not req.issubset(set(df.columns)):
        missing = req - set(df.columns)
        return OhlcvValidationReport(False, len(df), [f"missing columns: {sorted(missing)}"])

    t = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if t.isna().any():
        issues.append("some ts could not be parsed (NaT)")

    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")

    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        issues.append("non-positive OHLC values")

    bad_hl = (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
    if bad_hl.any():
        issues.append(f"OHLC inconsistent high/low vs open/close: {int(bad_hl.sum())} bars")

    if not t.is_monotonic_increasing:
        issues.append("ts not sorted ascending")

    dup = t.duplicated().sum()
    if dup:
        issues.append(f"duplicate ts: {int(dup)}")

    return OhlcvValidationReport(len(issues) == 0, len(df), issues)
