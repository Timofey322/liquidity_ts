from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

# OANDA (брокер) — REST v20. Токен личного доступа в кабинете OANDA.
# Practice: api-fxpractice.oanda.com | Live: api-fxtrade.oanda.com


@dataclass
class OandaConfig:
    practice: bool = False
    timeout_sec: float = 60.0


def _host(cfg: OandaConfig) -> str:
    return "https://api-fxpractice.oanda.com" if cfg.practice else "https://api-fxtrade.oanda.com"


def _token() -> str:
    t = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_TOKEN") or ""
    if not t.strip():
        raise RuntimeError(
            "Задайте переменную окружения OANDA_API_TOKEN (Personal Access Token в кабинете OANDA). "
            "Для practice используйте --practice и токен practice-аккаунта."
        )
    return t.strip()


def _iso_z(ts: pd.Timestamp) -> str:
    s = ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    return s


def fetch_candles_range(
    instrument: str,
    granularity: str,
    start: datetime,
    end: datetime,
    *,
    cfg: OandaConfig | None = None,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Свечи по **mid** (OANDA): ts, open, high, low, close, volume.

    instrument: EUR_USD, GBP_USD, XAU_USD, ...
    granularity: S5, M1, M5, M15, M30, H1, H4, D, ...
    """
    cfg = cfg or OandaConfig()
    tok = token or _token()
    host = _host(cfg)
    url = f"{host}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {tok}"}

    cursor = pd.Timestamp(start)
    if cursor.tzinfo is None:
        cursor = cursor.tz_localize(timezone.utc)
    else:
        cursor = cursor.tz_convert(timezone.utc)

    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(timezone.utc)
    else:
        end_ts = end_ts.tz_convert(timezone.utc)

    if cursor >= end_ts:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    rows: List[dict] = []
    max_pages = 5000
    for _ in range(max_pages):
        if cursor >= end_ts:
            break
        params = {
            "granularity": granularity,
            "price": "M",
            "from": _iso_z(cursor),
            "to": _iso_z(end_ts),
        }
        r = requests.get(url, headers=headers, params=params, timeout=cfg.timeout_sec)
        r.raise_for_status()
        candles = r.json().get("candles") or []
        if not candles:
            break

        last_complete: pd.Timestamp | None = None
        for c in candles:
            if not c.get("complete", True):
                continue
            ts = pd.to_datetime(c["time"], utc=True)
            if ts < cursor or ts >= end_ts:
                continue
            mid = c["mid"]
            rows.append(
                {
                    "ts": ts,
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c.get("volume", 0) or 0),
                }
            )
            last_complete = ts

        if last_complete is None:
            break
        nxt = last_complete + pd.Timedelta(milliseconds=1)
        if nxt <= cursor:
            break
        cursor = nxt

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def to_pair_tag(instrument: str) -> str:
    return instrument.replace("_", "").upper()
