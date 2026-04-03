from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.mtf_reversal import (
    ResearchConfig,
    _build_dataset,
    _fetch_aggregated_bars,
    _resample_ohlcv,
    _train_model,
)


PAIR_FEE_BPS = {
    "EURUSD": 1.0,
    "GBPUSD": 1.0,
    "XAUUSD": 2.0,
}

PAIR_SESSION_FILTERS = {
    "EURUSD": (6, 18),
    "GBPUSD": (6, 18),
    "XAUUSD": (12, 21),
}


SCENARIOS = [
    {
        "name": "1h_5m",
        "base_rule": "5min",
        "htf_rule": "1h",
        "idea_window_bars": 24,  # 2 часа окна после HTF-идеи
        "max_hold_bars": 48,  # 4 часа удержания
    },
    {
        "name": "4h_15m",
        "base_rule": "15min",
        "htf_rule": "4h",
        "idea_window_bars": 8,  # те же 2 часа в абсолютном времени
        "max_hold_bars": 16,  # те же 4 часа удержания
    },
]


def _load_or_fetch_bars_5m(pair: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pair.lower()}_bars_5m.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["ts"])

    # Для EURUSD уже есть готовый полугодовой прогон 1H -> 5M.
    if pair == "EURUSD":
        existing = Path("models/research_eurusd_halfyear_5m1h/bars_5m.csv")
        if existing.exists():
            df = pd.read_csv(existing, parse_dates=["ts"])
            df.to_csv(cache_path, index=False)
            return df

    cfg = ResearchConfig(pair=pair, start=start, end=end, base_rule="5min", out_dir=str(cache_dir / f"{pair.lower()}_tmp"))
    df = _fetch_aggregated_bars(cfg)
    df.to_csv(cache_path, index=False)
    return df


def _scenario_cfg(pair: str, start: str, end: str, out_dir: Path, sc: dict) -> ResearchConfig:
    session_start, session_end = PAIR_SESSION_FILTERS[pair]
    atr_window = 288 if sc["base_rule"] == "5min" else 96
    return ResearchConfig(
        pair=pair,
        start=start,
        end=end,
        out_dir=str(out_dir),
        base_rule=sc["base_rule"],
        htf_rule=sc["htf_rule"],
        chunk_days=2,
        fractal_left=1,
        fractal_right=1,
        idea_window_bars=int(sc["idea_window_bars"]),
        max_hold_bars=int(sc["max_hold_bars"]),
        fee_bps=float(PAIR_FEE_BPS[pair]),
        n_trials=10,
        train_ratio=0.6,
        val_ratio=0.2,
        min_prob_threshold=0.45,
        max_prob_threshold=0.85,
        footprint_z_window=20,
        session_start_hour_utc=session_start,
        session_end_hour_utc=session_end,
        atr_quantile_window=atr_window,
        min_atr_quantile=0.25 if pair == "XAUUSD" else 0.35,
        max_atr_quantile=0.98,
        random_state=42,
    )


def main() -> None:
    start = "2024-01-01T00:00:00"
    end = "2024-07-01T00:00:00"
    compare_root = Path("models/compare_halfyear")
    cache_dir = compare_root / "cache"
    compare_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for pair in ("EURUSD", "GBPUSD", "XAUUSD"):
        print(f"[PAIR] {pair}")
        bars_5m = _load_or_fetch_bars_5m(pair, start, end, cache_dir)
        bars_15m = _resample_ohlcv(bars_5m, "15min")
        bars_map = {
            "5min": bars_5m,
            "15min": bars_15m,
        }

        for sc in SCENARIOS:
            out_dir = compare_root / f"{pair.lower()}_{sc['name']}"
            cfg = _scenario_cfg(pair, start, end, out_dir, sc)
            bars = bars_map[cfg.base_rule].copy()
            out_dir.mkdir(parents=True, exist_ok=True)
            bars.to_csv(out_dir / "bars.csv", index=False)
            dataset = _build_dataset(bars, cfg)
            dataset.to_csv(out_dir / "dataset.csv", index=False)
            meta = _train_model(dataset, cfg, out_dir)
            print(f"[DONE] {pair} {sc['name']} test_return={meta['test_total_return']:.4f}")

            summary_rows.append(
                {
                    "pair": pair,
                    "scenario": sc["name"],
                    "htf_rule": sc["htf_rule"],
                    "base_rule": sc["base_rule"],
                    "fee_bps": meta["config"]["fee_bps"],
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
            )

    summary = pd.DataFrame(summary_rows).sort_values(["pair", "scenario"]).reset_index(drop=True)
    summary.to_csv(compare_root / "summary.csv", index=False)
    (compare_root / "summary.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
