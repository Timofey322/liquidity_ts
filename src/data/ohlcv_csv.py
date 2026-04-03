from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_histdata_like_csv(
    path: str | Path,
    *,
    sep: str = ";",
    decimal: str = ".",
    has_header: bool = False,
) -> pd.DataFrame:
    """
    Типичный экспорт HistData: Date Time OHLC(V) без заголовка, разделитель ';'.

    Пример строки:
      20000103 000500;1.36697;1.36719;1.36666;1.36691;0
    """
    p = Path(path)
    df = pd.read_csv(p, sep=sep, header=None if not has_header else 0, engine="python")
    if df.shape[1] < 5:
        raise ValueError(f"Expected >=5 columns (datetime+OHLC), got {df.shape[1]}")
    # колонки 0 = datetime string split or combined
    if df.shape[1] >= 6:
        dt = df[0].astype(str).str.strip() + " " + df[1].astype(str).str.strip()
        o, h, l, c = df[2], df[3], df[4], df[5]
        vol = df[6] if df.shape[1] > 6 else 0
    else:
        dt = df[0].astype(str)
        o, h, l, c = df[1], df[2], df[3], df[4]
        vol = df[5] if df.shape[1] > 5 else 0
    ts = pd.to_datetime(dt, format="%Y%m%d %H%M%S", utc=True, errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(dt, utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "ts": ts,
            "open": pd.to_numeric(o.astype(str).str.replace(",", decimal, regex=False), errors="coerce"),
            "high": pd.to_numeric(h.astype(str).str.replace(",", decimal, regex=False), errors="coerce"),
            "low": pd.to_numeric(l.astype(str).str.replace(",", decimal, regex=False), errors="coerce"),
            "close": pd.to_numeric(c.astype(str).str.replace(",", decimal, regex=False), errors="coerce"),
            "volume": pd.to_numeric(vol, errors="coerce").fillna(0.0),
        }
    )
    return out.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)


def load_generic_ohlcv_csv(
    path: str | Path,
    *,
    ts_col: str = "ts",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    sep: str = ",",
) -> pd.DataFrame:
    """Универсальный CSV (например экспорт с Forex.com / MT5): задайте имена колонок."""
    df = pd.read_csv(path, sep=sep)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df.get(volume_col, 0.0), errors="coerce").fillna(0.0),
        }
    )
    return out.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
