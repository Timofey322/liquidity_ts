from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.research.mtf_reversal import ResearchConfig, _build_dataset, _resample_ohlcv, _train_model


CANDIDATES = [
    {
        "name": "xau_1h5m_baseline",
        "base_rule": "5min",
        "htf_rule": "1h",
        "idea_window_bars": 24,
        "max_hold_bars": 48,
        "use_sweep_idea": True,
        "use_fvg_idea": True,
        "footprint_threshold": 0.25,
        "entry_mode": "cross_or_ret",
        "stop_atr_mult": 1.0,
        "target_rr": 1.5,
    },
    {
        "name": "xau_1h5m_sweep_strict",
        "base_rule": "5min",
        "htf_rule": "1h",
        "idea_window_bars": 18,
        "max_hold_bars": 72,
        "use_sweep_idea": True,
        "use_fvg_idea": False,
        "footprint_threshold": 0.5,
        "entry_mode": "cross_only",
        "stop_atr_mult": 1.2,
        "target_rr": 2.0,
    },
    {
        "name": "xau_1h5m_sweep_and_ret",
        "base_rule": "5min",
        "htf_rule": "1h",
        "idea_window_bars": 18,
        "max_hold_bars": 72,
        "use_sweep_idea": True,
        "use_fvg_idea": False,
        "footprint_threshold": 0.75,
        "entry_mode": "cross_and_ret",
        "stop_atr_mult": 1.0,
        "target_rr": 1.8,
    },
    {
        "name": "xau_1h5m_mixed_strict",
        "base_rule": "5min",
        "htf_rule": "1h",
        "idea_window_bars": 24,
        "max_hold_bars": 60,
        "use_sweep_idea": True,
        "use_fvg_idea": True,
        "footprint_threshold": 0.5,
        "entry_mode": "cross_and_ret",
        "stop_atr_mult": 1.2,
        "target_rr": 2.0,
    },
    {
        "name": "xau_4h15m_baseline",
        "base_rule": "15min",
        "htf_rule": "4h",
        "idea_window_bars": 8,
        "max_hold_bars": 16,
        "use_sweep_idea": True,
        "use_fvg_idea": True,
        "footprint_threshold": 0.25,
        "entry_mode": "cross_or_ret",
        "stop_atr_mult": 1.0,
        "target_rr": 1.5,
    },
    {
        "name": "xau_4h15m_sweep_strict",
        "base_rule": "15min",
        "htf_rule": "4h",
        "idea_window_bars": 12,
        "max_hold_bars": 24,
        "use_sweep_idea": True,
        "use_fvg_idea": False,
        "footprint_threshold": 0.5,
        "entry_mode": "cross_only",
        "stop_atr_mult": 1.2,
        "target_rr": 2.0,
    },
    {
        "name": "xau_4h15m_sweep_and_ret",
        "base_rule": "15min",
        "htf_rule": "4h",
        "idea_window_bars": 12,
        "max_hold_bars": 24,
        "use_sweep_idea": True,
        "use_fvg_idea": False,
        "footprint_threshold": 0.75,
        "entry_mode": "cross_and_ret",
        "stop_atr_mult": 1.0,
        "target_rr": 1.8,
    },
    {
        "name": "xau_4h15m_mixed_strict",
        "base_rule": "15min",
        "htf_rule": "4h",
        "idea_window_bars": 12,
        "max_hold_bars": 20,
        "use_sweep_idea": True,
        "use_fvg_idea": True,
        "footprint_threshold": 0.5,
        "entry_mode": "cross_and_ret",
        "stop_atr_mult": 1.2,
        "target_rr": 2.0,
    },
]


def main() -> None:
    root = Path("models/xau_tuning")
    root.mkdir(parents=True, exist_ok=True)

    bars_5m = pd.read_csv("models/compare_halfyear/cache/xauusd_bars_5m.csv", parse_dates=["ts"])
    bars_15m = _resample_ohlcv(bars_5m, "15min")
    bars_map = {"5min": bars_5m, "15min": bars_15m}

    rows: list[dict] = []
    best_row: dict | None = None

    for cand in CANDIDATES:
        out_dir = root / cand["name"]
        cfg = ResearchConfig(
            pair="XAUUSD",
            start="2024-01-01T00:00:00",
            end="2024-07-01T00:00:00",
            out_dir=str(out_dir),
            chunk_days=2,
            fee_bps=2.0,
            n_trials=12,
            train_ratio=0.6,
            val_ratio=0.2,
            fetch_retries=3,
            fetch_timeout_sec=120.0,
            session_start_hour_utc=12,
            session_end_hour_utc=21,
            atr_quantile_window=288 if cand["base_rule"] == "5min" else 96,
            min_atr_quantile=0.25,
            max_atr_quantile=0.98,
            random_state=42,
            **{k: v for k, v in cand.items() if k != "name"},
        )
        dataset = _build_dataset(bars_map[cfg.base_rule].copy(), cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(out_dir / "dataset.csv", index=False)
        meta = _train_model(dataset, cfg, out_dir)
        row = {
            "name": cand["name"],
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
        rows.append(row)
        if best_row is None or row["best_val_score"] > best_row["best_val_score"]:
            best_row = deepcopy(row)
        print(
            f"[DONE] {cand['name']} val_score={row['best_val_score']:.4f} "
            f"test_return={row['test_total_return']:.4f}"
        )

    summary = pd.DataFrame(rows).sort_values(["best_val_score", "test_total_return"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(root / "summary.csv", index=False)
    (root / "summary.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    (root / "best_config.json").write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
