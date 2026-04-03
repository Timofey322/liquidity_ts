from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.accounting.ledger import build_close_report
from src.accounting.strategy_trades import convert_strategy_trades_file


INSTRUMENTS = {
    "XAUUSD": "models/session_poi_xau_tuned/best_run/trades_all.csv",
    "GBPUSD": "models/session_poi/gbpusd/trades.csv",
    "EURUSD": "models/session_poi/eurusd/trades.csv",
}


def _equity_from_cash_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    out = df[["ts", "cash_balance"]].copy()
    out.columns = ["ts", "equity"]
    return out


def main() -> None:
    root = Path(".")
    out_root = root / "models" / "accounting" / "three_instruments"
    out_root.mkdir(parents=True, exist_ok=True)

    per_inst_eq: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for symbol, rel_path in INSTRUMENTS.items():
        in_path = root / rel_path
        if not in_path.exists():
            raise FileNotFoundError(f"Input trades not found for {symbol}: {in_path}")
        trades = convert_strategy_trades_file(
            str(in_path),
            symbol=symbol,
            quantity=1.0,
            fee_per_fill=0.0,
        )
        ext = pd.DataFrame([{"symbol": symbol, "quantity": 0.0}])
        out_dir = out_root / symbol.lower()
        summary = build_close_report(trades, out_dir, external_positions_df=ext)
        per_inst_eq[symbol] = _equity_from_cash_ledger(out_dir / "cash_ledger.csv")
        summary_rows.append({"symbol": symbol} | summary)

    # Combined equity as sum of per-instrument cash curves aligned by time.
    all_ts = pd.Index([])
    for df in per_inst_eq.values():
        all_ts = all_ts.union(pd.Index(df["ts"]))
    all_ts = pd.DatetimeIndex(all_ts).sort_values()
    combined = pd.DataFrame({"ts": all_ts})
    combined["equity"] = 0.0
    for symbol, df in per_inst_eq.items():
        s = df.set_index("ts")["equity"].reindex(all_ts).ffill().fillna(0.0)
        combined["equity"] += s.to_numpy(dtype=float)

    # Plot separate + combined
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"XAUUSD": "#d62728", "GBPUSD": "#1f77b4", "EURUSD": "#2ca02c"}
    for symbol, df in per_inst_eq.items():
        ax.plot(df["ts"], df["equity"], label=symbol, linewidth=1.6, alpha=0.9, color=colors.get(symbol))
    ax.plot(combined["ts"], combined["equity"], label="COMBINED", linewidth=2.2, color="#111111")
    ax.set_title("Accounting equity by instrument and combined")
    ax.set_ylabel("Equity (cash balance)")
    ax.grid(alpha=0.25)
    ax.legend()
    chart_path = out_root / "equity_by_instrument_and_combined.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)

    combined.to_csv(out_root / "combined_equity.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_root / "summary_by_instrument.csv", index=False)
    (out_root / "summary_by_instrument.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {chart_path}")
    print(f"saved: {out_root / 'summary_by_instrument.csv'}")


if __name__ == "__main__":
    main()

