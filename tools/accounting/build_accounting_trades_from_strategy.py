from __future__ import annotations

import argparse
from pathlib import Path

from src.accounting.strategy_trades import convert_strategy_trades_file


def convert_strategy_trades(path: str, symbol: str, qty: float, fee_per_fill: float) -> pd.DataFrame:
    return convert_strategy_trades_file(path, symbol=symbol, quantity=qty, fee_per_fill=fee_per_fill)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert strategy trades to accounting ledger format")
    p.add_argument("--in-csv", required=True, help="Strategy trades CSV with entry_ts/exit_ts/side/entry/exit_price")
    p.add_argument("--out-csv", required=True, help="Output accounting trades CSV")
    p.add_argument("--symbol", default="XAUUSD", help="Instrument symbol to write")
    p.add_argument("--quantity", type=float, default=1.0, help="Units per trade leg")
    p.add_argument("--fee-per-fill", type=float, default=0.0, help="Absolute fee per fill")
    ns = p.parse_args()

    out = convert_strategy_trades(ns.in_csv, ns.symbol, ns.quantity, ns.fee_per_fill)
    out_path = Path(ns.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()

