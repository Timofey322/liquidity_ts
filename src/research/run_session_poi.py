from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.session_poi_strategy import SessionPoiConfig, run_session_poi


CONFIGS = [
    SessionPoiConfig(pair="EURUSD", bars_path="models/compare_halfyear/cache/eurusd_bars_5m.csv", out_dir="models/session_poi/eurusd", fee_bps=1.0),
    SessionPoiConfig(pair="GBPUSD", bars_path="models/compare_halfyear/cache/gbpusd_bars_5m.csv", out_dir="models/session_poi/gbpusd", fee_bps=1.0),
    SessionPoiConfig(pair="XAUUSD", bars_path="models/compare_halfyear/cache/xauusd_bars_5m.csv", out_dir="models/session_poi/xauusd", fee_bps=2.0),
]


def main() -> None:
    rows = []
    for cfg in CONFIGS:
        summary = run_session_poi(cfg)
        rows.append(
            {
                "pair": summary["pair"],
                "n_trades": summary["n_trades"],
                "total_return": summary["total_return"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": summary["win_rate"],
                "avg_trade": summary["avg_trade"],
                "avg_r_multiple": summary["avg_r_multiple"],
            }
        )
    df = pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)
    root = Path("models/session_poi")
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "summary.csv", index=False)
    (root / "summary.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
