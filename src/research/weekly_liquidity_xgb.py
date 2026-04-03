from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as e:  # pragma: no cover
    optuna = None
    _OPTUNA_ERR = e
else:
    _OPTUNA_ERR = None

try:
    from xgboost import XGBClassifier
except ImportError as e:  # pragma: no cover
    XGBClassifier = None
    _XGB_ERR = e
else:
    _XGB_ERR = None


@dataclass
class WeeklyXgbConfig:
    pair: str
    bars_path: str
    out_dir: str
    fee_bps: float = 2.0
    train_weeks: int = 2
    move_horizon_bars: int = 36
    max_signals_per_week: int = 8
    min_event_move: float = 0.0015
    big_move_quantile: float = 0.85
    val_ratio: float = 0.25
    n_trials: int = 14
    random_state: int = 42
    kernel_window: int = 48
    kernel_bandwidth: float = 14.0
    sl_atr_quantile: float = 0.75
    tp_atr_quantile: float = 0.55
    min_sl_atr: float = 0.7
    max_sl_atr: float = 3.0
    min_tp_atr: float = 1.0
    max_tp_atr: float = 6.0


def _load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    req = {"ts", "open", "high", "low", "close"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)


def _week_start_utc(ts: pd.Series) -> pd.Series:
    day = ts.dt.floor("D")
    return day - pd.to_timedelta(ts.dt.weekday, unit="D")


def _atr(df: pd.DataFrame, span: int = 14) -> np.ndarray:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr).ewm(span=span, adjust=False).mean().to_numpy()


def _kernel_regression_past(series: np.ndarray, window: int, bandwidth: float) -> np.ndarray:
    n = series.size
    out = np.full(n, np.nan, dtype=np.float64)
    w = int(max(window, 3))
    bw = max(float(bandwidth), 1.0)
    dist = np.arange(w - 1, -1, -1, dtype=np.float64)  # 0 for current point, large for old points
    weights = np.exp(-0.5 * (dist / bw) ** 2)
    weights = weights / (weights.sum() + 1e-12)
    for i in range(w - 1, n):
        x = series[i - w + 1 : i + 1]
        if np.all(np.isfinite(x)):
            out[i] = float(np.sum(x * weights))
    return out


def _build_features(df: pd.DataFrame, cfg: WeeklyXgbConfig) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    open_ = out["open"].to_numpy(dtype=np.float64)
    vol = out["volume"].to_numpy(dtype=np.float64)

    out["atr"] = _atr(out)
    out["ret1"] = pd.Series(close).pct_change(fill_method=None).fillna(0.0)
    out["ret3"] = pd.Series(close).pct_change(3, fill_method=None).fillna(0.0)
    out["ret6"] = pd.Series(close).pct_change(6, fill_method=None).fillna(0.0)
    out["atr_rel"] = out["atr"] / (out["close"] + 1e-12)
    out["range_rel"] = (high - low) / (close + 1e-12)
    out["body_rel"] = (close - open_) / (out["atr"] + 1e-12)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    out["upper_wick_rel"] = upper_wick / (out["atr"] + 1e-12)
    out["lower_wick_rel"] = lower_wick / (out["atr"] + 1e-12)
    out["vol_log1p"] = np.log1p(np.maximum(vol, 0.0))

    ts = pd.to_datetime(out["ts"], utc=True)
    out["hour"] = ts.dt.hour.astype(np.int16)
    out["dow"] = ts.dt.weekday.astype(np.int16)

    # Session context (UTC): Asia 00-07, London 07-16, New York 13-22.
    hour = ts.dt.hour
    day = ts.dt.floor("D")
    in_asia = ((hour >= 0) & (hour < 7)).to_numpy()
    in_london = ((hour >= 7) & (hour < 16)).to_numpy()
    in_ny = ((hour >= 13) & (hour < 22)).to_numpy()
    out["in_session_asia"] = in_asia.astype(np.float64)
    out["in_session_london"] = in_london.astype(np.float64)
    out["in_session_ny"] = in_ny.astype(np.float64)
    out["in_session_london_ny_overlap"] = (in_london & in_ny).astype(np.float64)

    # Build previous-day session extremes (fully known, no lookahead leak).
    sess = pd.Series("off", index=out.index, dtype=object)
    sess[in_asia] = "asia"
    sess[in_london & (~in_ny)] = "london"
    sess[in_ny] = "ny"
    sess_df = pd.DataFrame(
        {
            "day": day,
            "session": sess,
            "high": out["high"].to_numpy(dtype=np.float64),
            "low": out["low"].to_numpy(dtype=np.float64),
        }
    )
    sess_df = sess_df[sess_df["session"] != "off"]
    if not sess_df.empty:
        daily_levels = (
            sess_df.groupby(["day", "session"], sort=False)
            .agg(session_high=("high", "max"), session_low=("low", "min"))
            .reset_index()
        )
        daily_levels["day"] = pd.to_datetime(daily_levels["day"], utc=True, errors="coerce")
        wide_hi = daily_levels.pivot(index="day", columns="session", values="session_high")
        wide_lo = daily_levels.pivot(index="day", columns="session", values="session_low")
        wide_hi.columns = [f"{c}_high" for c in wide_hi.columns]
        wide_lo.columns = [f"{c}_low" for c in wide_lo.columns]
        levels = wide_hi.join(wide_lo, how="outer").sort_index()
        prev_levels = levels.shift(1).rename(columns=lambda c: f"prev_{c}")
        out["_day_utc"] = day
        out = out.merge(prev_levels, left_on="_day_utc", right_index=True, how="left")
        out = out.drop(columns=["_day_utc"])
    else:
        out["prev_asia_high"] = np.nan
        out["prev_asia_low"] = np.nan
        out["prev_london_high"] = np.nan
        out["prev_london_low"] = np.nan
        out["prev_ny_high"] = np.nan
        out["prev_ny_low"] = np.nan

    for col in [
        "prev_asia_high",
        "prev_asia_low",
        "prev_london_high",
        "prev_london_low",
        "prev_ny_high",
        "prev_ny_low",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    atr_eps = out["atr"].to_numpy(dtype=np.float64) + 1e-12
    close_arr = out["close"].to_numpy(dtype=np.float64)
    high_arr = out["high"].to_numpy(dtype=np.float64)
    low_arr = out["low"].to_numpy(dtype=np.float64)
    close_prev = np.roll(close_arr, 1)
    close_prev[0] = close_arr[0]

    for name in ("asia", "london", "ny"):
        hi_col = f"prev_{name}_high"
        lo_col = f"prev_{name}_low"
        hi = out[hi_col].to_numpy(dtype=np.float64)
        lo = out[lo_col].to_numpy(dtype=np.float64)

        out[f"dist_prev_{name}_high_atr"] = (close_arr - hi) / atr_eps
        out[f"dist_prev_{name}_low_atr"] = (close_arr - lo) / atr_eps

        cross_up = np.isfinite(hi) & (close_arr >= hi) & (close_prev < hi)
        cross_down = np.isfinite(lo) & (close_arr <= lo) & (close_prev > lo)
        touch_hi = np.isfinite(hi) & (high_arr >= hi)
        touch_lo = np.isfinite(lo) & (low_arr <= lo)
        sweep_reclaim_hi = np.isfinite(hi) & (high_arr > hi) & (close_arr < hi)
        sweep_reclaim_lo = np.isfinite(lo) & (low_arr < lo) & (close_arr > lo)

        out[f"cross_prev_{name}_high"] = cross_up.astype(np.float64)
        out[f"cross_prev_{name}_low"] = cross_down.astype(np.float64)
        out[f"touch_prev_{name}_high"] = touch_hi.astype(np.float64)
        out[f"touch_prev_{name}_low"] = touch_lo.astype(np.float64)
        out[f"sweep_reclaim_prev_{name}_high"] = sweep_reclaim_hi.astype(np.float64)
        out[f"sweep_reclaim_prev_{name}_low"] = sweep_reclaim_lo.astype(np.float64)

    # Kernel-regression feature block (requested "регрессивное ядро")
    kr = _kernel_regression_past(close, window=cfg.kernel_window, bandwidth=cfg.kernel_bandwidth)
    out["kr_close"] = kr
    out["kr_resid"] = (close - kr) / (close + 1e-12)
    out["kr_slope_3"] = (pd.Series(kr).diff(3) / 3.0).to_numpy(dtype=np.float64) / (close + 1e-12)
    out["kr_slope_8"] = (pd.Series(kr).diff(8) / 8.0).to_numpy(dtype=np.float64) / (close + 1e-12)
    out["kr_curve"] = pd.Series(kr).diff(1).diff(1).to_numpy(dtype=np.float64) / (close + 1e-12)

    # Rolling context
    out["roll_ret_std_24"] = out["ret1"].rolling(24, min_periods=8).std(ddof=0)
    out["roll_range_med_24"] = out["range_rel"].rolling(24, min_periods=8).median()
    out["roll_atr_rel_med_48"] = out["atr_rel"].rolling(48, min_periods=12).median()
    return out


def _future_moves(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    atr = df["atr"].to_numpy(dtype=np.float64)
    n = len(df)
    up = np.zeros(n, dtype=np.float64)
    dn = np.zeros(n, dtype=np.float64)
    mae_long = np.zeros(n, dtype=np.float64)
    mfe_long = np.zeros(n, dtype=np.float64)
    mae_short = np.zeros(n, dtype=np.float64)
    mfe_short = np.zeros(n, dtype=np.float64)
    for i in range(n):
        j1 = i + 1
        j2 = min(i + 1 + int(horizon), n)
        if j1 >= j2 or c[i] <= 0 or atr[i] <= 0:
            continue
        fut_hi = float(np.max(h[j1:j2]))
        fut_lo = float(np.min(l[j1:j2]))
        up[i] = max(0.0, (fut_hi / c[i]) - 1.0)
        dn[i] = max(0.0, 1.0 - (fut_lo / c[i]))
        mae_long[i] = max(0.0, (c[i] - fut_lo) / (atr[i] + 1e-12))
        mfe_long[i] = max(0.0, (fut_hi - c[i]) / (atr[i] + 1e-12))
        mae_short[i] = max(0.0, (fut_hi - c[i]) / (atr[i] + 1e-12))
        mfe_short[i] = max(0.0, (c[i] - fut_lo) / (atr[i] + 1e-12))
    out = df.copy()
    out["up_move"] = up
    out["dn_move"] = dn
    out["move_abs"] = np.maximum(up, dn)
    out["mae_long_atr"] = mae_long
    out["mfe_long_atr"] = mfe_long
    out["mae_short_atr"] = mae_short
    out["mfe_short_atr"] = mfe_short
    return out


def _labels_for_window(win: pd.DataFrame, cfg: WeeklyXgbConfig) -> tuple[np.ndarray, float]:
    thr = max(float(win["move_abs"].quantile(cfg.big_move_quantile)), float(cfg.min_event_move))
    y = np.zeros(len(win), dtype=np.int64)  # 0 neutral, 1 long_big, 2 short_big
    long_big = (win["up_move"] >= win["dn_move"]) & (win["move_abs"] >= thr)
    short_big = (win["dn_move"] > win["up_move"]) & (win["move_abs"] >= thr)
    y[long_big.to_numpy()] = 1
    y[short_big.to_numpy()] = 2
    return y, float(thr)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {
        "ts",
        "open",
        "high",
        "low",
        "close",
        "week_start",
        "up_move",
        "dn_move",
        "move_abs",
        "mae_long_atr",
        "mfe_long_atr",
        "mae_short_atr",
        "mfe_short_atr",
    }
    return [c for c in df.columns if c not in drop]


def build_weekly_xgb_training_matrix(
    bars_path: str | Path,
    cfg: WeeklyXgbConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Полный датафрейм признаков + метки для weekly XGB (без weekly walk-forward).
    Возвращает (X, y, feature_names, meta).
    """
    path = Path(bars_path)
    if cfg is None:
        cfg = WeeklyXgbConfig(pair="GENERIC", bars_path=str(path), out_dir="")
    df = _load_bars(str(path))
    df = _build_features(df, cfg)
    df = _future_moves(df, int(cfg.move_horizon_bars))
    y, move_thr = _labels_for_window(df, cfg)
    feat = _feature_cols(df)
    valid = np.isfinite(df[feat].to_numpy(dtype=np.float64)).all(axis=1)
    valid &= np.isfinite(df["move_abs"].to_numpy(dtype=np.float64))
    valid_idx = np.where(valid)[0]
    X = df.loc[valid, feat].to_numpy(dtype=np.float32)
    y_fit = y[valid_idx]
    meta = {
        "bars_path": str(path.resolve()),
        "n_rows_raw": int(len(df)),
        "n_rows_fit": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "move_threshold": float(move_thr),
        "class_counts": {
            int(u): int(c)
            for u, c in zip(*np.unique(y_fit, return_counts=True))
        },
    }
    return X, y_fit, feat, meta


def _score_bt(total_return: float, mdd: float, n_trades: int) -> float:
    if n_trades <= 0:
        return -1e9
    trade_factor = min(float(n_trades), 10.0) / 10.0
    return float(trade_factor * total_return / (1.0 + abs(mdd)))


def _simulate_one(week_df: pd.DataFrame, i: int, side: int, sl_atr: float, tp_atr: float, hold_bars: int) -> tuple[int, float, float, str]:
    c = week_df["close"].to_numpy(dtype=np.float64)
    h = week_df["high"].to_numpy(dtype=np.float64)
    l = week_df["low"].to_numpy(dtype=np.float64)
    atr = week_df["atr"].to_numpy(dtype=np.float64)
    entry = float(c[i])
    atr0 = float(atr[i])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr0) or atr0 <= 0:
        return i + 1, entry, entry, "invalid"
    if side > 0:
        stop = entry - sl_atr * atr0
        target = entry + tp_atr * atr0
    else:
        stop = entry + sl_atr * atr0
        target = entry - tp_atr * atr0
    last = min(i + int(hold_bars), len(week_df) - 1)
    exit_idx = last
    exit_px = float(c[last])
    reason = "time"
    for j in range(i + 1, last + 1):
        if side > 0:
            if l[j] <= stop:
                exit_idx, exit_px, reason = j, stop, "stop"
                break
            if h[j] >= target:
                exit_idx, exit_px, reason = j, target, "target"
                break
        else:
            if h[j] >= stop:
                exit_idx, exit_px, reason = j, stop, "stop"
                break
            if l[j] <= target:
                exit_idx, exit_px, reason = j, target, "target"
                break
    return exit_idx, float(stop), float(exit_px), reason


def _backtest_by_proba(
    week_df: pd.DataFrame,
    proba: np.ndarray,
    *,
    prob_threshold: float,
    sl_atr: float,
    tp_atr: float,
    fee_bps: float,
    move_horizon_bars: int,
    max_signals_per_week: int,
) -> pd.DataFrame:
    if week_df.empty or proba.size == 0:
        return pd.DataFrame()
    p_long = proba[:, 1]
    p_short = proba[:, 2]
    p_big = np.maximum(p_long, p_short)
    order = np.argsort(-p_big)

    trades: list[dict] = []
    next_free = 0
    used = 0
    ts = pd.to_datetime(week_df["ts"], utc=True)
    fee = float(fee_bps) / 10000.0
    for idx in order:
        i = int(idx)
        if p_big[i] < float(prob_threshold):
            break
        if i < next_free:
            continue
        if used >= int(max_signals_per_week):
            break
        side = 1 if p_long[i] >= p_short[i] else -1
        exit_idx, stop, exit_px, reason = _simulate_one(week_df, i, side, sl_atr, tp_atr, move_horizon_bars)
        entry = float(week_df["close"].iloc[i])
        if not np.isfinite(exit_px) or exit_px <= 0 or entry <= 0:
            continue
        gross = (exit_px / entry) - 1.0 if side > 0 else 1.0 - (exit_px / entry)
        net = float(gross) - fee
        trades.append(
            {
                "entry_ts": ts.iloc[i],
                "exit_ts": ts.iloc[exit_idx],
                "side": "long" if side > 0 else "short",
                "entry": entry,
                "stop": stop,
                "exit_price": float(exit_px),
                "exit_reason": reason,
                "p_long": float(p_long[i]),
                "p_short": float(p_short[i]),
                "p_big": float(p_big[i]),
                "gross_ret": float(gross),
                "net_ret": float(net),
            }
        )
        next_free = exit_idx + 1
        used += 1
    return pd.DataFrame(trades).sort_values("entry_ts").reset_index(drop=True) if trades else pd.DataFrame()


def _summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "total_return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "sharpe": 0.0}
    arr = trades["net_ret"].to_numpy(dtype=np.float64)
    eq = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    sharpe = 0.0
    if arr.size > 1:
        sd = float(np.std(arr, ddof=1))
        if sd > 1e-12:
            sharpe = float((float(np.mean(arr)) / sd) * np.sqrt(float(arr.size)))
    return {
        "n_trades": int(arr.size),
        "total_return": float(eq[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "win_rate": float(np.mean(arr > 0)),
        "avg_trade": float(np.mean(arr)),
        "sharpe": float(sharpe),
    }


def _optimize_xgb(train_df: pd.DataFrame, cfg: WeeklyXgbConfig) -> tuple[dict, float, float, float]:
    if optuna is None:
        raise RuntimeError("optuna is required for optimization") from _OPTUNA_ERR
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for optimization") from _XGB_ERR

    feat_cols = _feature_cols(train_df)
    y, _ = _labels_for_window(train_df, cfg)
    valid = np.isfinite(train_df[feat_cols]).all(axis=1).to_numpy()
    valid &= np.isfinite(train_df["move_abs"].to_numpy())
    trdf = train_df.loc[valid].reset_index(drop=True)
    y = y[valid]
    if len(trdf) < 300:
        raise ValueError("Not enough rows for xgb weekly training window.")

    n = len(trdf)
    n_val = max(int(n * cfg.val_ratio), 120)
    n_val = min(n_val, n - 120)
    X = trdf[feat_cols].to_numpy(dtype=np.float32)
    X_tr = X[: n - n_val]
    y_tr = y[: n - n_val]
    X_va = X[n - n_val :]
    va_df = trdf.iloc[n - n_val :].reset_index(drop=True)
    y_tr_classes = np.unique(y_tr)
    if y_tr_classes.size < 2:
        raise ValueError("Training window has only one class.")

    # Side-aware risk from historical big moves in train part.
    long_big = (y_tr == 1)
    short_big = (y_tr == 2)
    sl_long = np.quantile(trdf["mae_long_atr"].iloc[: n - n_val][long_big], cfg.sl_atr_quantile) if np.any(long_big) else 1.2
    tp_long = np.quantile(trdf["mfe_long_atr"].iloc[: n - n_val][long_big], cfg.tp_atr_quantile) if np.any(long_big) else 1.8
    sl_short = np.quantile(trdf["mae_short_atr"].iloc[: n - n_val][short_big], cfg.sl_atr_quantile) if np.any(short_big) else sl_long
    tp_short = np.quantile(trdf["mfe_short_atr"].iloc[: n - n_val][short_big], cfg.tp_atr_quantile) if np.any(short_big) else tp_long
    sl_atr = float(np.clip(np.nanmean([sl_long, sl_short]), cfg.min_sl_atr, cfg.max_sl_atr))
    tp_atr = float(np.clip(np.nanmean([tp_long, tp_short]), cfg.min_tp_atr, cfg.max_tp_atr))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 8.0, log=True),
            "random_state": cfg.random_state,
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
            "n_jobs": 1,
        }
        prob_threshold = trial.suggest_float("prob_threshold", 0.35, 0.80)
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)
        trades = _backtest_by_proba(
            va_df,
            proba,
            prob_threshold=prob_threshold,
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            fee_bps=cfg.fee_bps,
            move_horizon_bars=cfg.move_horizon_bars,
            max_signals_per_week=max(4, cfg.max_signals_per_week),
        )
        s = _summary(trades)
        score = _score_bt(float(s["total_return"]), float(s["max_drawdown"]), int(s["n_trades"]))
        trial.set_user_attr("val_return", float(s["total_return"]))
        trial.set_user_attr("val_dd", float(s["max_drawdown"]))
        trial.set_user_attr("val_trades", int(s["n_trades"]))
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg.random_state)),
    )
    study.optimize(objective, n_trials=int(cfg.n_trials), show_progress_bar=False)
    best = dict(study.best_params)
    best_threshold = float(best.pop("prob_threshold"))
    best["objective"] = "multi:softprob"
    best["num_class"] = 3
    best["tree_method"] = "hist"
    best["n_jobs"] = 1
    best["random_state"] = cfg.random_state
    return best, best_threshold, sl_atr, tp_atr


def run_weekly_xgb(cfg: WeeklyXgbConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _build_features(_load_bars(cfg.bars_path), cfg)
    df = _future_moves(df, cfg.move_horizon_bars)
    df["week_start"] = _week_start_utc(pd.to_datetime(df["ts"], utc=True))
    weeks = sorted(df["week_start"].dropna().unique().tolist())
    if len(weeks) <= int(cfg.train_weeks):
        raise ValueError("Not enough weeks for weekly xgb run.")

    all_trades: list[pd.DataFrame] = []
    weekly_rows: list[dict] = []
    feat_cols = _feature_cols(df)

    for i in range(int(cfg.train_weeks), len(weeks)):
        train_weeks = weeks[i - int(cfg.train_weeks) : i]
        test_week = weeks[i]
        train_df = df[df["week_start"].isin(train_weeks)].copy().reset_index(drop=True)
        test_df = df[df["week_start"] == test_week].copy().reset_index(drop=True)
        if len(test_df) < 20:
            continue
        try:
            best_params, threshold, sl_atr, tp_atr = _optimize_xgb(train_df, cfg)
        except Exception as exc:
            weekly_rows.append({"week_start": str(test_week), "status": f"skip:{type(exc).__name__}"})
            continue

        y_train, thr_move = _labels_for_window(train_df, cfg)
        train_valid = np.isfinite(train_df[feat_cols]).all(axis=1).to_numpy()
        train_valid &= np.isfinite(train_df["move_abs"].to_numpy())
        X_train = train_df.loc[train_valid, feat_cols].to_numpy(dtype=np.float32)
        y_fit = y_train[train_valid]
        if np.unique(y_fit).size < 2:
            weekly_rows.append({"week_start": str(test_week), "status": "skip:one_class_fit"})
            continue
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_fit)

        test_valid = np.isfinite(test_df[feat_cols]).all(axis=1).to_numpy()
        td = test_df.loc[test_valid].reset_index(drop=True)
        if td.empty:
            weekly_rows.append({"week_start": str(test_week), "status": "skip:no_test_rows"})
            continue
        X_test = td[feat_cols].to_numpy(dtype=np.float32)
        proba = model.predict_proba(X_test)
        trades = _backtest_by_proba(
            td,
            proba,
            prob_threshold=threshold,
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            fee_bps=cfg.fee_bps,
            move_horizon_bars=cfg.move_horizon_bars,
            max_signals_per_week=cfg.max_signals_per_week,
        )
        if not trades.empty:
            trades["week_start"] = str(test_week)
            all_trades.append(trades)
        weekly_rows.append(
            {
                "week_start": str(test_week),
                "status": "ok",
                "move_threshold": float(thr_move),
                "prob_threshold": float(threshold),
                "sl_atr": float(sl_atr),
                "tp_atr": float(tp_atr),
                "rr": float(tp_atr / (sl_atr + 1e-12)),
                "n_trades_week": int(len(trades)),
            }
        )

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = _summary(trades_all)
    summary = {"pair": cfg.pair, "config": asdict(cfg)} | summary

    if not trades_all.empty:
        eq = pd.Series(
            np.cumprod(1.0 + trades_all["net_ret"].to_numpy(dtype=np.float64)),
            index=pd.to_datetime(trades_all["entry_ts"], utc=True),
        )
        eq.to_csv(out / "equity_curve.csv", index=True, header=["equity"])
        trades_all.to_csv(out / "trades.csv", index=False)
    else:
        pd.Series([1.0], dtype=np.float64).to_csv(out / "equity_curve.csv", index=False, header=["equity"])
    pd.DataFrame(weekly_rows).to_csv(out / "weekly_xgb_meta.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> WeeklyXgbConfig:
    p = argparse.ArgumentParser(description="Weekly adaptive liquidity XGBoost strategy")
    p.add_argument("--pair", default="XAUUSD")
    p.add_argument("--bars-path", default="models/compare_halfyear/cache/xauusd_bars_5m.csv")
    p.add_argument("--out-dir", default="models/weekly_xgb/xauusd")
    p.add_argument("--fee-bps", type=float, default=2.0)
    p.add_argument("--train-weeks", type=int, default=2)
    p.add_argument("--move-horizon-bars", type=int, default=36)
    p.add_argument("--max-signals-per-week", type=int, default=8)
    p.add_argument("--min-event-move", type=float, default=0.0015)
    p.add_argument("--big-move-quantile", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--n-trials", type=int, default=14)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--kernel-window", type=int, default=48)
    p.add_argument("--kernel-bandwidth", type=float, default=14.0)
    ns = p.parse_args()
    return WeeklyXgbConfig(**vars(ns))


if __name__ == "__main__":
    result = run_weekly_xgb(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
