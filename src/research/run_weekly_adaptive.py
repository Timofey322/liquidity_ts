from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.weekly_liquidity_adaptive import WeeklyAdaptiveConfig, run_weekly_adaptive


CONFIGS = [
    WeeklyAdaptiveConfig(
        pair="EURUSD",
        bars_path="models/compare_halfyear/cache/eurusd_bars_5m.csv",
        out_dir="models/weekly_adaptive/eurusd",
        fee_bps=1.0,
    ),
    WeeklyAdaptiveConfig(
        pair="GBPUSD",
        bars_path="models/compare_halfyear/cache/gbpusd_bars_5m.csv",
        out_dir="models/weekly_adaptive/gbpusd",
        fee_bps=1.0,
    ),
    WeeklyAdaptiveConfig(
        pair="XAUUSD",
        bars_path="models/compare_halfyear/cache/xauusd_bars_5m.csv",
        out_dir="models/weekly_adaptive/xauusd",
        fee_bps=2.0,
    ),
]


def main() -> None:
    rows: list[dict] = []
    for cfg in CONFIGS:
        s = run_weekly_adaptive(cfg)
        rows.append(
            {
                "pair": s["pair"],
                "n_trades": s["n_trades"],
                "total_return": s["total_return"],
                "max_drawdown": s["max_drawdown"],
                "win_rate": s["win_rate"],
                "avg_trade": s["avg_trade"],
                "sharpe": s["sharpe"],
            }
        )

    out = Path("models/weekly_adaptive")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)
    df.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
