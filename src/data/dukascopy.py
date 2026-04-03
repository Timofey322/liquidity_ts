from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pandas as pd
import requests

# Публичный архив котировок Dukascopy (не поток брокера для исполнения ордеров).
BASE = "https://datafeed.dukascopy.com/datafeed"


@dataclass
class DukascopyConfig:
    timeout_sec: float = 60.0
    user_agent: str = "Mozilla/5.0 (compatible; research/1.0)"


def _hour_url(pair: str, dt_utc: datetime) -> str:
    pair = pair.upper().replace("/", "")
    y = dt_utc.year
    m0 = dt_utc.month - 1  # 0..11 в пути Dukascopy
    d = dt_utc.day
    h = dt_utc.hour
    return f"{BASE}/{pair}/{y}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5"


def _parse_hour_ticks(raw: bytes, hour_start: datetime) -> List[Tuple[datetime, float, float, float, float]]:
    if not raw:
        return []
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return []
    out: List[Tuple[datetime, float, float, float, float]] = []
    t0 = hour_start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for i in range(0, len(data), 20):
        chunk = data[i : i + 20]
        if len(chunk) < 20:
            break
        # Формат Dukascopy .bi5 (20 байт, big-endian):
        # uint32 ms от начала часа; int32 bid/ask в пунктах * 1e5; float ask_vol; float bid_vol
        ms, bid_i, ask_i, av, bv = struct.unpack(">Iii ff", chunk)
        ts = t0 + timedelta(milliseconds=int(ms))
        bid = float(bid_i) / 100_000.0
        ask = float(ask_i) / 100_000.0
        out.append((ts, bid, ask, float(av), float(bv)))
    return out


def fetch_ticks_hour(pair: str, dt_utc: datetime, cfg: DukascopyConfig | None = None) -> List[Tuple[datetime, float, float, float, float]]:
    cfg = cfg or DukascopyConfig()
    url = _hour_url(pair, dt_utc)
    r = requests.get(
        url,
        timeout=cfg.timeout_sec,
        headers={"User-Agent": cfg.user_agent},
    )
    if r.status_code != 200:
        return []
    return _parse_hour_ticks(r.content, dt_utc)


def fetch_ticks_range(
    pair: str,
    start: datetime,
    end: datetime,
    cfg: DukascopyConfig | None = None,
) -> List[Tuple[datetime, float, float, float, float]]:
    """
    Скачивает тики по часам в диапазоне [start, end) UTC.
    """
    cfg = cfg or DukascopyConfig()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    cur = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_utc = end.astimezone(timezone.utc)
    all_ticks: List[Tuple[datetime, float, float, float, float]] = []
    while cur < end_utc:
        all_ticks.extend(fetch_ticks_hour(pair, cur, cfg=cfg))
        cur += timedelta(hours=1)
    all_ticks.sort(key=lambda x: x[0])
    return all_ticks


def ticks_to_dataframe(ticks: List[Tuple[datetime, float, float, float, float]]) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame(columns=["ts", "bid", "ask", "ask_vol", "bid_vol"])
    df = pd.DataFrame(ticks, columns=["ts", "bid", "ask", "ask_vol", "bid_vol"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df
