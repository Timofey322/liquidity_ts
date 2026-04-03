from __future__ import annotations

import pandas as pd

STRATEGY_REQUIRED = {"entry_ts", "exit_ts", "side", "entry", "exit_price"}


def convert_strategy_trades_to_ledger(
    df: pd.DataFrame,
    *,
    symbol: str,
    quantity: float,
    fee_per_fill: float,
) -> pd.DataFrame:
    miss = STRATEGY_REQUIRED - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in strategy trades: {sorted(miss)}")

    side = df["side"].astype(str).str.lower().str.strip()
    entry_is_buy = side.eq("long")
    sym = symbol.upper().strip()
    q = float(quantity)
    fee = float(fee_per_fill)

    rows: list[dict] = []
    for r, is_buy in zip(df.itertuples(index=False), entry_is_buy, strict=False):
        entry_side = "buy" if is_buy else "sell"
        exit_side = "sell" if is_buy else "buy"
        rows.append(
            {
                "ts": r.entry_ts,
                "symbol": sym,
                "side": entry_side,
                "quantity": q,
                "price": float(r.entry),
                "fee": fee,
            }
        )
        rows.append(
            {
                "ts": r.exit_ts,
                "symbol": sym,
                "side": exit_side,
                "quantity": q,
                "price": float(r.exit_price),
                "fee": fee,
            }
        )
    out = pd.DataFrame(rows)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out


def convert_strategy_trades_file(
    path: str,
    *,
    symbol: str,
    quantity: float,
    fee_per_fill: float,
) -> pd.DataFrame:
    return convert_strategy_trades_to_ledger(pd.read_csv(path), symbol=symbol, quantity=quantity, fee_per_fill=fee_per_fill)
