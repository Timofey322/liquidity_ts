from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_trade: float
    equity_curve: pd.Series


def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12)
    return float(dd.min())


def _sharpe(returns: np.ndarray) -> float:
    """Sharpe-подобная метрика по серии сделок (без жёсткой годовой нормы)."""
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * np.sqrt(float(returns.size))


def score_backtest(bt: BacktestResult) -> float:
    """
    Целевая функция под реальную торговую задачу:
    максимизируем net return, но штрафуем сильную просадку и слишком редкие сделки.
    """
    if bt.n_trades == 0:
        return -1e9
    trade_factor = min(float(bt.n_trades), 10.0) / 10.0
    return float(trade_factor * bt.total_return / (1.0 + abs(bt.max_drawdown)))


def backtest_from_predictions(
    dataset: pd.DataFrame,
    proba: np.ndarray,
    *,
    fee_bps: float = 1.0,
    prob_threshold: float = 0.35,
    hold_bars: int = 1,
    allow_overlap: bool = False,
) -> BacktestResult:
    """
    Бэктест по уже рассчитанным вероятностям модели.
    По умолчанию не допускаем перекрывающиеся сделки: одна позиция держится `hold_bars`.
    """
    if len(dataset) != int(proba.shape[0]):
        raise ValueError("dataset/proba length mismatch")
    if "close_exit" not in dataset.columns:
        raise ValueError("Dataset must contain close_exit.")

    fee = float(fee_bps) / 10000.0  # roundtrip fee on simple returns
    hold_bars = max(int(hold_bars), 1)
    rets: List[float] = []
    eq_index: List[pd.Timestamp] = []
    next_free_idx = 0

    for k in range(len(dataset)):
        if (not allow_overlap) and k < next_free_idx:
            continue
        p = proba[k]
        p_long = float(p[1]) if p.shape[0] > 1 else 0.0
        p_short = float(p[2]) if p.shape[0] > 2 else 0.0
        if max(p_long, p_short) < float(prob_threshold):
            continue
        cls = 1 if p_long >= p_short else 2
        c0 = float(dataset["close"].iloc[k])
        c1 = float(dataset["close_exit"].iloc[k])
        if c0 <= 0:
            continue
        r = (c1 / c0) - 1.0
        r_net = (r - fee) if cls == 1 else ((-r) - fee)
        rets.append(r_net)
        eq_index.append(pd.to_datetime(dataset["ts"].iloc[k], utc=True))
        if not allow_overlap:
            next_free_idx = k + hold_bars

    if not rets:
        eq = pd.Series([1.0], dtype=np.float64)
        return BacktestResult(0.0, 0.0, 0.0, 0, 0.0, 0.0, eq)

    r = np.array(rets, dtype=np.float64)
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1] - 1.0)
    win_rate = float(np.mean(r > 0))
    sharpe = _sharpe(r)
    mdd = _max_dd(equity)
    return BacktestResult(
        total_return=total,
        sharpe=sharpe,
        max_drawdown=mdd,
        n_trades=int(len(r)),
        win_rate=win_rate,
        avg_trade=float(np.mean(r)),
        equity_curve=pd.Series(equity, index=eq_index),
    )


def backtest_long_short(
    dataset: pd.DataFrame,
    model_dir: str | Path,
    *,
    fee_bps: float = 1.0,
    prob_threshold: float = -1.0,
    allow_overlap: bool = False,
) -> BacktestResult:
    """
    Бэктест на отложенной выборке (ts >= test_ts_start из meta.json).
    Сигнал: argmax prob; торгуем long при классе 1, short при 2, если prob > threshold.
    Если threshold < 0, берём `best_prob_threshold` из meta.json.
    PnL: простая доходность от close[i] до close[i+horizon] с учётом fee_bps на вход+выход.
    """
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    mdir = Path(model_dir)
    meta = json.loads((mdir / "meta.json").read_text(encoding="utf-8"))
    test_ts_start = pd.to_datetime(meta["test_ts_start"], utc=True)
    feat_cols = meta["feature_cols"]
    clf: HistGradientBoostingClassifier = joblib.load(mdir / "model.joblib")
    scaler = joblib.load(mdir / "scaler.joblib")
    hold_bars = int(meta.get("horizon", 1))
    if float(prob_threshold) < 0:
        prob_threshold = float(meta.get("best_prob_threshold", 0.35))

    df = dataset.sort_values("ts").reset_index(drop=True)
    if "close_exit" not in df.columns:
        raise ValueError("Dataset must contain close_exit (rebuild with prepare-ml-dataset).")
    test = df[df["ts"] >= test_ts_start].copy()
    if test.empty:
        raise ValueError("No rows in dataset for test period (check test_ts_start vs dataset ts).")

    X = test[feat_cols].to_numpy(dtype=np.float64)
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)
    return backtest_from_predictions(
        test,
        proba,
        fee_bps=float(fee_bps),
        prob_threshold=float(prob_threshold),
        hold_bars=hold_bars,
        allow_overlap=allow_overlap,
    )
