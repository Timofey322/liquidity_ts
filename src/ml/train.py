from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.ml.backtest import backtest_from_predictions, score_backtest
from src.ml.dataset import feature_columns

try:
    import optuna
except ImportError as e:  # pragma: no cover
    optuna = None
    _OPTUNA_ERR = e
else:
    _OPTUNA_ERR = None


@dataclass
class TrainConfig:
    n_trials: int = 40
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    # test = rest (0.2)
    random_state: int = 42
    min_rows: int = 80
    fee_bps: float = 1.0
    min_prob_threshold: float = 0.35
    max_prob_threshold: float = 0.75
    allow_overlap: bool = False


def _time_splits(
    df: pd.DataFrame, cfg: TrainConfig, *, purge_bars: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, Any]:
    d = df.sort_values("ts").reset_index(drop=True)
    n = len(d)
    purge_bars = max(int(purge_bars), 0)
    n_eff = n - 2 * purge_bars
    if n_eff <= 0:
        raise ValueError(f"Too few rows for split after purge gap: n={n}, purge={purge_bars}")
    n_tr = int(n_eff * cfg.train_ratio)
    n_va = int(n_eff * cfg.val_ratio)
    n_te = n_eff - n_tr - n_va
    if n_te < 10 or n_tr < 30:
        raise ValueError(
            f"Too few rows for split: n={n} purge={purge_bars} "
            f"(train={n_tr}, val={n_va}, test={n_te})"
        )
    tr = d.iloc[:n_tr].copy()
    va_start = n_tr + purge_bars
    va = d.iloc[va_start : va_start + n_va].copy()
    te_start = va_start + n_va + purge_bars
    te = d.iloc[te_start : te_start + n_te].copy()
    test_ts_start = te["ts"].iloc[0]
    return tr, va, te, test_ts_start, d


def train_with_optuna(
    dataset: pd.DataFrame,
    out_dir: str | Path,
    cfg: TrainConfig | None = None,
    *,
    horizon: int = 12,
    ret_threshold: float = 0.0003,
) -> Dict[str, Any]:
    if optuna is None:
        raise RuntimeError("optuna is required: pip install optuna") from _OPTUNA_ERR

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = cfg or TrainConfig()
    if len(dataset) < cfg.min_rows:
        raise ValueError(f"Dataset too small: {len(dataset)} < {cfg.min_rows}")

    tr, va, te, test_ts_start, full = _time_splits(dataset, cfg, purge_bars=int(horizon))
    feat_cols = feature_columns(dataset)
    X_tr, y_tr = tr[feat_cols].to_numpy(dtype=np.float64), tr["y"].to_numpy(dtype=np.int64)
    X_va, y_va = va[feat_cols].to_numpy(dtype=np.float64), va["y"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "random_state": cfg.random_state,
        }
        clf = HistGradientBoostingClassifier(class_weight="balanced", **params)
        clf.fit(X_trs, y_tr)
        prob_threshold = trial.suggest_float("prob_threshold", cfg.min_prob_threshold, cfg.max_prob_threshold)
        proba = clf.predict_proba(X_vas)
        bt = backtest_from_predictions(
            va,
            proba,
            fee_bps=float(cfg.fee_bps),
            prob_threshold=float(prob_threshold),
            hold_bars=int(horizon),
            allow_overlap=bool(cfg.allow_overlap),
        )
        score = score_backtest(bt)
        trial.set_user_attr("val_total_return", float(bt.total_return))
        trial.set_user_attr("val_sharpe", float(bt.sharpe))
        trial.set_user_attr("val_max_drawdown", float(bt.max_drawdown))
        trial.set_user_attr("val_n_trades", int(bt.n_trades))
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg.random_state)),
    )
    study.optimize(objective, n_trials=int(cfg.n_trials), show_progress_bar=False)

    best = study.best_params
    best_prob_threshold = float(best.pop("prob_threshold"))
    best_clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        max_iter=best["max_iter"],
        min_samples_leaf=best["min_samples_leaf"],
        l2_regularization=best["l2_regularization"],
        random_state=cfg.random_state,
    )
    # финальное обучение: train + val (скейлер тоже переобучаем только на fit-сегменте)
    X_fit_raw = np.vstack([X_tr, X_va])
    y_fit = np.concatenate([y_tr, y_va])
    fit_scaler = StandardScaler()
    X_fit = fit_scaler.fit_transform(X_fit_raw)
    best_clf.fit(X_fit, y_fit)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(best_clf, out / "model.joblib")
    joblib.dump(fit_scaler, out / "scaler.joblib")

    best_trial = study.best_trial

    meta = {
        "feature_cols": feat_cols,
        "best_params": best,
        "best_prob_threshold": best_prob_threshold,
        "best_val_score": float(study.best_value),
        "best_val_total_return": float(best_trial.user_attrs.get("val_total_return", 0.0)),
        "best_val_sharpe": float(best_trial.user_attrs.get("val_sharpe", 0.0)),
        "best_val_max_drawdown": float(best_trial.user_attrs.get("val_max_drawdown", 0.0)),
        "best_val_n_trades": int(best_trial.user_attrs.get("val_n_trades", 0)),
        "objective_metric": "return_over_drawdown",
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "purge_bars": int(horizon),
        "fee_bps": float(cfg.fee_bps),
        "allow_overlap": bool(cfg.allow_overlap),
        "test_ts_start": str(test_ts_start),
        "horizon": int(horizon),
        "ret_threshold": float(ret_threshold),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_full": int(len(full)),
        "class_counts": {str(k): int(v) for k, v in dataset["y"].value_counts().items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out),
        "best_val_score": float(study.best_value),
        "best_prob_threshold": best_prob_threshold,
        "test_ts_start": str(test_ts_start),
    }
