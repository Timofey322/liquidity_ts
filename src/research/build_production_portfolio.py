from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.research.mtf_reversal import ResearchConfig, _build_dataset, _train_model


PRODUCTION_CONFIGS = [
    {
        "pair": "EURUSD",
        "name": "eurusd_prod_4h15m",
        "base_rule": "15min",
        "htf_rule": "4h",
        "use_sweep_idea": True,
        "use_fvg_idea": True,
        "footprint_threshold": 0.25,
        "entry_mode": "cross_or_ret",
        "stop_atr_mult": 1.0,
        "target_rr": 1.5,
        "idea_window_bars": 8,
        "max_hold_bars": 16,
        "session_start_hour_utc": 6,
        "session_end_hour_utc": 18,
        "atr_quantile_window": 96,
        "min_atr_quantile": 0.35,
        "max_atr_quantile": 0.98,
        "fee_bps": 1.0,
        "n_trials": 40,
    },
    {
        "pair": "GBPUSD",
        "name": "gbpusd_prod_4h15m",
        "base_rule": "15min",
        "htf_rule": "4h",
        "use_sweep_idea": True,
        "use_fvg_idea": True,
        "footprint_threshold": 0.25,
        "entry_mode": "cross_or_ret",
        "stop_atr_mult": 1.0,
        "target_rr": 1.5,
        "idea_window_bars": 8,
        "max_hold_bars": 16,
        "session_start_hour_utc": 6,
        "session_end_hour_utc": 18,
        "atr_quantile_window": 96,
        "min_atr_quantile": 0.35,
        "max_atr_quantile": 0.98,
        "fee_bps": 1.0,
        "n_trials": 40,
    },
]


def _stability_score(row: dict) -> float:
    val_ret = float(row["best_val_total_return"])
    test_ret = float(row["test_total_return"])
    test_mdd = abs(float(row["test_max_drawdown"]))
    test_sharpe = max(float(row["test_sharpe"]), 0.0)
    trades = int(row["test_n_trades"])

    if test_ret <= 0:
        return 0.0
    consistency = 0.0
    if val_ret > 0:
        consistency = min(val_ret, test_ret) / max(val_ret, test_ret)
    trade_factor = min(trades, 100) / 100.0
    return float(consistency * (test_ret / (1.0 + test_mdd)) * (1.0 + test_sharpe / 2.0) * (0.5 + 0.5 * trade_factor))


def main() -> None:
    root = Path("models/production_portfolio")
    root.mkdir(parents=True, exist_ok=True)
    cache = Path("models/compare_halfyear/cache")

    rows: list[dict] = []
    equities: dict[str, pd.Series] = {}

    for cfg_raw in PRODUCTION_CONFIGS:
        pair = cfg_raw["pair"]
        bars_path = cache / f"{pair.lower()}_bars_5m.csv"
        bars = pd.read_csv(bars_path, parse_dates=["ts"])
        out_dir = root / cfg_raw["name"]
        cfg = ResearchConfig(
            pair=pair,
            start="2024-01-01T00:00:00",
            end="2024-07-01T00:00:00",
            out_dir=str(out_dir),
            chunk_days=2,
            fetch_retries=3,
            fetch_timeout_sec=120.0,
            train_ratio=0.6,
            val_ratio=0.2,
            random_state=42,
            **{k: v for k, v in cfg_raw.items() if k not in {"pair", "name"}},
        )
        dataset = _build_dataset(bars.copy(), cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(out_dir / "dataset.csv", index=False)
        meta = _train_model(dataset, cfg, out_dir)
        row = {
            "pair": pair,
            "config_name": cfg_raw["name"],
            "base_rule": cfg.base_rule,
            "htf_rule": cfg.htf_rule,
            "use_sweep_idea": cfg.use_sweep_idea,
            "use_fvg_idea": cfg.use_fvg_idea,
            "footprint_threshold": cfg.footprint_threshold,
            "entry_mode": cfg.entry_mode,
            "stop_atr_mult": cfg.stop_atr_mult,
            "target_rr": cfg.target_rr,
            "idea_window_bars": cfg.idea_window_bars,
            "max_hold_bars": cfg.max_hold_bars,
            "n_trials": cfg.n_trials,
            "fee_bps": cfg.fee_bps,
            "best_prob_threshold": meta["best_prob_threshold"],
            "best_val_score": meta["best_val_score"],
            "best_val_total_return": meta["best_val_total_return"],
            "best_val_sharpe": meta["best_val_sharpe"],
            "best_val_max_drawdown": meta["best_val_max_drawdown"],
            "best_val_n_trades": meta["best_val_n_trades"],
            "test_total_return": meta["test_total_return"],
            "test_sharpe": meta["test_sharpe"],
            "test_max_drawdown": meta["test_max_drawdown"],
            "test_n_trades": meta["test_n_trades"],
            "test_win_rate": meta["test_win_rate"],
            "test_avg_trade": meta["test_avg_trade"],
            "n_train": meta["n_train"],
            "n_val": meta["n_val"],
            "n_test": meta["n_test"],
            "class_0": meta["class_counts"].get("0", 0),
            "class_1": meta["class_counts"].get("1", 0),
            "class_2": meta["class_counts"].get("2", 0),
            "test_ts_start": meta["test_ts_start"],
        }
        row["stability_score"] = _stability_score(row)
        rows.append(row)
        eq = pd.read_csv(out_dir / "equity_curve.csv")
        equities[pair] = pd.Series(eq.iloc[:, 1].to_numpy(), index=pd.to_datetime(eq.iloc[:, 0], utc=True, errors="coerce"))
        print(
            f"[DONE] {pair} test_return={row['test_total_return']:.4f} "
            f"test_sharpe={row['test_sharpe']:.3f} stability={row['stability_score']:.6f}"
        )

    summary = pd.DataFrame(rows).sort_values("stability_score", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    summary.to_csv(root / "summary.csv", index=False)
    (root / "summary.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    best_configs = {
        row["pair"]: {
            "config_name": row["config_name"],
            "stability_score": row["stability_score"],
            "test_total_return": row["test_total_return"],
            "test_sharpe": row["test_sharpe"],
            "test_max_drawdown": row["test_max_drawdown"],
            "best_prob_threshold": row["best_prob_threshold"],
        }
        for row in summary.to_dict(orient="records")
    }
    (root / "best_configs.json").write_text(json.dumps(best_configs, ensure_ascii=False, indent=2), encoding="utf-8")

    chart_dir = root / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for pair, s in equities.items():
        plt.plot(s.index, s.to_numpy(), label=pair, linewidth=1.8)
    plt.title("Deployable production portfolio equity by pair")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "production_equity_by_pair.png", dpi=160)
    plt.close()

    aligned = pd.concat({k: v for k, v in equities.items()}, axis=1).ffill().dropna(how="all")
    combined = aligned.mean(axis=1)
    plt.figure(figsize=(12, 4.8))
    plt.plot(combined.index, combined.to_numpy(), color="black", linewidth=2.0)
    plt.title("Equal-weight combined deployable equity")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(chart_dir / "production_equity_combined.png", dpi=160)
    plt.close()

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
