from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.session_poi_strategy import (
    SessionPoiConfig,
    build_session_poi_dataset,
    plot_trade_examples,
    simulate_session_poi_trades,
    summarize_trades,
)


def _score(summary: dict) -> float:
    n_trades = float(summary.get("n_trades", 0))
    if n_trades <= 0:
        return -1e9
    trade_factor = min(n_trades, 100.0) / 100.0
    total_return = float(summary.get("total_return", 0.0))
    max_dd = abs(float(summary.get("max_drawdown", 0.0)))
    return trade_factor * total_return / (1.0 + max_dd)


def _split_masks(df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> dict[str, np.ndarray]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = max(n - n_train - n_val, 0)
    idx = np.arange(n)
    masks = {
        "train": idx < n_train,
        "val": (idx >= n_train) & (idx < (n_train + n_val)),
        "test": (idx >= (n_train + n_val)) & (idx < (n_train + n_val + n_test)),
        "all": np.ones(n, dtype=bool),
    }
    return masks


def _run_split(df: pd.DataFrame, cfg: SessionPoiConfig, mask: np.ndarray) -> tuple[pd.DataFrame, pd.Series, dict]:
    part = df.loc[mask].reset_index(drop=True).copy()
    trades, eq = simulate_session_poi_trades(part, cfg)
    summary = summarize_trades(trades, eq)
    return trades, eq, summary


def _random_config(rng: np.random.Generator) -> SessionPoiConfig:
    session_variants = [
        ("london", "newyork"),
        ("asia", "london", "newyork"),
        ("newyork",),
        ("london",),
    ]
    sess_idx = int(rng.integers(0, len(session_variants)))
    return SessionPoiConfig(
        pair="XAUUSD",
        bars_path="models/compare_halfyear/cache/xauusd_bars_5m.csv",
        out_dir="models/session_poi_xau_tuned/best_run",
        fee_bps=2.0,
        poi_window_bars=int(rng.choice([8, 12, 18, 24, 30])),
        body_threshold=float(rng.choice([0.35, 0.5, 0.6, 0.75, 0.9, 1.1])),
        stop_buffer_atr=float(rng.choice([0.05, 0.1, 0.15, 0.2])),
        target_rr=float(rng.choice([2.0])),
        max_hold_bars=int(rng.choice([24, 36, 48, 72, 96])),
        session_fractal_left=int(rng.choice([1, 2])),
        session_fractal_right=int(rng.choice([1, 2])),
        ltf_fractal_left=int(rng.choice([1, 2])),
        ltf_fractal_right=int(rng.choice([1, 2])),
        allowed_sessions=session_variants[sess_idx],
        use_fractal_sweep_poi=True,
        use_fvg_poi=True,
        use_bos_entry=bool(rng.integers(0, 2)),
        use_choch_entry=False,
        use_inversion_entry=False,
        require_trend_filter=True,
        trend_lookback_bars=int(rng.choice([6, 12, 18, 24])),
        ema_span=int(rng.choice([34, 50, 89, 144])),
    )


def main() -> None:
    out_root = Path("models/session_poi_xau_tuned")
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_trials = 220

    rows: list[dict] = []
    best: dict | None = None
    best_cfg: SessionPoiConfig | None = None
    best_df: pd.DataFrame | None = None

    for trial in range(n_trials):
        cfg = _random_config(rng)
        if not (cfg.use_fractal_sweep_poi or cfg.use_fvg_poi):
            cfg.use_fractal_sweep_poi = True
        if not (cfg.use_bos_entry or cfg.use_choch_entry or cfg.use_inversion_entry):
            cfg.use_bos_entry = True

        df = build_session_poi_dataset(cfg)
        masks = _split_masks(df)
        _, _, train_s = _run_split(df, cfg, masks["train"])
        _, _, val_s = _run_split(df, cfg, masks["val"])
        _, _, test_s = _run_split(df, cfg, masks["test"])
        val_score = _score(val_s)
        row = {
            "trial": trial,
            "val_score": val_score,
            "train_total_return": train_s["total_return"],
            "val_total_return": val_s["total_return"],
            "test_total_return": test_s["total_return"],
            "train_max_drawdown": train_s["max_drawdown"],
            "val_max_drawdown": val_s["max_drawdown"],
            "test_max_drawdown": test_s["max_drawdown"],
            "train_n_trades": train_s["n_trades"],
            "val_n_trades": val_s["n_trades"],
            "test_n_trades": test_s["n_trades"],
            "config": json.dumps(asdict(cfg), ensure_ascii=False),
        }
        rows.append(row)

        if best is None or val_score > float(best["val_score"]):
            best = row
            best_cfg = cfg
            best_df = df.copy()
            print(f"[best] trial={trial} val_score={val_score:.6f} val_ret={val_s['total_return']:.4f}")

    trials_df = pd.DataFrame(rows).sort_values(["val_score", "test_total_return"], ascending=[False, False]).reset_index(drop=True)
    trials_df.to_csv(out_root / "trials.csv", index=False)

    if best_cfg is None or best_df is None or best is None:
        raise RuntimeError("No best config found.")

    split_masks = _split_masks(best_df)
    best_dir = out_root / "best_run"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(best_dir / "dataset.csv", index=False)

    final_summary: dict[str, dict] = {"config": asdict(best_cfg)}
    for split_name in ("train", "val", "test", "all"):
        trades, eq, summary = _run_split(best_df, best_cfg, split_masks[split_name])
        trades.to_csv(best_dir / f"trades_{split_name}.csv", index=False)
        eq.to_csv(best_dir / f"equity_{split_name}.csv", index=True, header=["equity"])
        final_summary[split_name] = summary
        if split_name == "all":
            plot_trade_examples(best_df, trades, best_dir / "charts", best_cfg.pair)

    (best_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "best_config.json").write_text(json.dumps(asdict(best_cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_summary["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
