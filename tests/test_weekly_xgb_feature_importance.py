"""
Проверка важности признаков weekly XGBoost-контура на максимально доступном локальном датасете.

Использует CSV с наибольшим размером на диске из models/compare_halfyear/cache/*_bars_5m.csv.
Результат сохраняется в models/feature_importance/weekly_xgb_last.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBClassifier

from src.research.weekly_liquidity_xgb import WeeklyXgbConfig, build_weekly_xgb_training_matrix


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _longest_bars_csv() -> Path:
    cache = _project_root() / "models" / "compare_halfyear" / "cache"
    candidates = list(cache.glob("*_bars_5m.csv"))
    if not candidates:
        pytest.skip(f"No OHLCV cache in {cache}")
    return max(candidates, key=lambda p: p.stat().st_size)


def test_weekly_xgb_feature_importance_and_save_report() -> None:
    bars = _longest_bars_csv()
    cfg = WeeklyXgbConfig(
        pair=bars.stem.replace("_bars_5m", "").upper(),
        bars_path=str(bars),
        out_dir="",
        fee_bps=1.0,
        move_horizon_bars=36,
        kernel_window=48,
        kernel_bandwidth=14.0,
        big_move_quantile=0.85,
        min_event_move=0.0015,
    )
    X, y, names, meta = build_weekly_xgb_training_matrix(bars, cfg)
    assert X.shape[0] > 500, "Недостаточно строк для устойчивой оценки"
    assert X.shape[1] == len(names)
    assert len(np.unique(y)) >= 2, "Нужны минимум 2 класса для XGB"

    clf = XGBClassifier(
        max_depth=6,
        learning_rate=0.08,
        n_estimators=200,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X, y)
    imp = clf.feature_importances_
    assert imp.shape[0] == len(names)
    assert float(np.sum(imp)) > 0

    ranked = sorted(zip(names, imp.tolist()), key=lambda x: -x[1])
    out_dir = _project_root() / "models" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_csv": str(bars.resolve()),
        "meta": meta,
        "feature_importance_gain_sklearn": [{"feature": n, "importance": float(v)} for n, v in ranked],
    }
    (out_dir / "weekly_xgb_last.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    top = ranked[0][0]
    assert isinstance(top, str) and len(top) > 0
