from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import print

from src.accounting.ledger import build_close_report
from src.data.dukascopy import DukascopyConfig, fetch_ticks_range, ticks_to_dataframe
from src.data.oanda import OandaConfig, fetch_candles_range, to_pair_tag
from src.data.ohlcv_csv import load_generic_ohlcv_csv, load_histdata_like_csv
from src.data.resample import ticks_to_ohlcv
from src.data.validate import validate_ohlcv
from src.logic.behavior import BehaviorConfig, analyze_session
from src.ml.backtest import backtest_long_short
from src.ml.dataset import DatasetConfig, build_liquidity_dataset
from src.ml.train import TrainConfig, train_with_optuna

app = typer.Typer(no_args_is_help=True)


def _exactly_one_ticks_or_ohlcv(ticks_csv: str, ohlcv_csv: str) -> None:
    h, o = bool(ticks_csv.strip()), bool(ohlcv_csv.strip())
    if h == o:
        print("[ERROR] укажите ровно один источник: --ticks-csv или --ohlcv-csv")
        raise typer.Exit(1)


def _load_ohlcv_file(path: str, histdata: bool):
    import pandas as pd

    if histdata:
        return load_histdata_like_csv(path)
    return load_generic_ohlcv_csv(path)


@app.command()
def init_data_dir(path: str = "data"):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / ".gitkeep").write_text("", encoding="utf-8")
    print(f"[OK] {p}")


@app.command("fetch-duka")
def fetch_duka(
    pair: str = typer.Option(..., help="EURUSD | GBPUSD | XAUUSD"),
    start: str = typer.Option(..., help="UTC ISO, например 2024-06-01T00:00:00"),
    end: str = typer.Option(..., help="UTC ISO, например 2024-06-01T06:00:00"),
    out_csv: str = typer.Option(..., help="Путь к CSV тиков"),
):
    """
    Тики из публичного архива Dukascopy (см. README — это не брокерский поток исполнения).
    """
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    ticks = fetch_ticks_range(pair, t0, t1, cfg=DukascopyConfig())
    df = ticks_to_dataframe(ticks)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] ticks saved {out_csv} rows={len(df)}")


@app.command()
def analyze(
    ticks_csv: str = typer.Option("", help="CSV тиков (ts,bid,ask,ask_vol,bid_vol)"),
    ohlcv_csv: str = typer.Option("", help="CSV свечей: ts,open,high,low,close[,volume]"),
    histdata: bool = typer.Option(False, help="--ohlcv-csv в формате HistData (;, без заголовка)"),
    pair: str = typer.Option(..., help="EURUSD | GBPUSD | XAUUSD"),
    rule: str = typer.Option("1min", help="Ресемпл OHLC из тиков (pandas offset); для --ohlcv-csv не используется"),
):
    """
    Фракталы + имбаланс (FVG) + футпринт/маркет-профиль (POC) + кандидаты TP/SL.
    """
    import pandas as pd

    _exactly_one_ticks_or_ohlcv(ticks_csv, ohlcv_csv)

    if ticks_csv.strip():
        df_ticks = pd.read_csv(ticks_csv, parse_dates=["ts"])
        df_ticks["ts"] = pd.to_datetime(df_ticks["ts"], utc=True, errors="coerce")
        df_ticks = df_ticks.dropna(subset=["ts"])
        if df_ticks.empty:
            print("[ERROR] empty ticks")
            raise typer.Exit(1)
        ohlc = ticks_to_ohlcv(df_ticks, rule=rule)
        if "ts" not in ohlc.columns and ohlc.index.name == "ts":
            ohlc = ohlc.reset_index()
        vr = validate_ohlcv(ohlc)
        if not vr.ok:
            print(f"[WARN] OHLCV validation: {vr.issues}")
    else:
        df_ticks = pd.DataFrame()
        ohlc = _load_ohlcv_file(ohlcv_csv.strip(), histdata)
        vr = validate_ohlcv(ohlc)
        if not vr.ok:
            print(f"[ERROR] OHLCV validation failed: {vr.issues}")
            raise typer.Exit(1)

    if ohlc.empty:
        print("[ERROR] empty OHLC")
        raise typer.Exit(1)

    cfg = BehaviorConfig()
    rep = analyze_session(ohlc, df_ticks, pair=pair, cfg=cfg)
    print(f"[bold]pair[/bold] {rep['pair']}  [bold]ts[/bold] {rep['last_ts']}  [bold]close[/bold] {rep['last_close']:.5f}")
    print(f"cross_up_fractal={rep['cross_up_fractal']} cross_dn_fractal={rep['cross_dn_fractal']} in_imbalance={rep['in_imbalance_zone']}")
    print(f"imbalance [{rep['imbalance_low']} .. {rep['imbalance_high']}]")
    print(f"footprint_session_delta={rep['footprint_session_delta']:.4f} poc={rep['poc']} dir={rep['direction_hint']}")
    for lv in rep["levels"]:
        print(f"  level {lv.kind} @ {lv.price:.5f}  strength={lv.strength}  ({lv.reason})")


@app.command("prepare-ml-dataset")
def prepare_ml_dataset(
    ticks_csv: str = typer.Option("", help="CSV тиков Dukascopy"),
    ohlcv_csv: str = typer.Option("", help="Готовые свечи OANDA/HistData/MT5"),
    histdata: bool = typer.Option(False, help="--ohlcv-csv в формате HistData"),
    pair: str = typer.Option(..., help="EURUSD | GBPUSD | XAUUSD"),
    out_csv: str = typer.Option("data/ml_dataset.csv", help="Выход: фичи + y + close/close_exit"),
    rule: str = typer.Option("1min", help="Базовый таймфрейм OHLC из тиков; для --ohlcv-csv не используется"),
    htf_rule: str = typer.Option("1h", help="Higher timeframe для мультитаймфрейм-признаков"),
    horizon: int = typer.Option(12, help="Горизонт метки (баров)"),
    ret_threshold: float = typer.Option(0.0003, help="Порог |forward ret| для класса up/down"),
    footprint_window: int = typer.Option(5, help="Окно суммирования футпринта (баров)"),
):
    import pandas as pd

    _exactly_one_ticks_or_ohlcv(ticks_csv, ohlcv_csv)

    if ticks_csv.strip():
        df_ticks = pd.read_csv(ticks_csv, parse_dates=["ts"])
        df_ticks["ts"] = pd.to_datetime(df_ticks["ts"], utc=True, errors="coerce")
        df_ticks = df_ticks.dropna(subset=["ts"])
        ohlc = ticks_to_ohlcv(df_ticks, rule=rule)
        if "ts" not in ohlc.columns:
            ohlc = ohlc.reset_index()
        ticks_for_ml = df_ticks
    else:
        ticks_for_ml = None
        ohlc = _load_ohlcv_file(ohlcv_csv.strip(), histdata)
        vr = validate_ohlcv(ohlc)
        if not vr.ok:
            print(f"[ERROR] OHLCV validation failed: {vr.issues}")
            raise typer.Exit(1)

    ohlc = ohlc.sort_values("ts").reset_index(drop=True)
    ds_cfg = DatasetConfig(
        horizon=int(horizon),
        ret_threshold=float(ret_threshold),
        footprint_window_bars=int(footprint_window),
        htf_rule=htf_rule,
    )
    ds = build_liquidity_dataset(ohlc, pair, ticks=ticks_for_ml, cfg=ds_cfg)
    if ds.empty:
        print("[ERROR] empty dataset (нужно больше тиков / другой диапазон)")
        raise typer.Exit(1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out_csv, index=False)
    print(f"[OK] ml dataset rows={len(ds)} cols={len(ds.columns)} -> {out_csv}")


@app.command("train-liquidity")
def train_liquidity(
    dataset_csv: str = typer.Option(..., help="CSV из prepare-ml-dataset"),
    out_dir: str = typer.Option("models/liquidity_hgb", help="Каталог: model.joblib, scaler.joblib, meta.json"),
    n_trials: int = typer.Option(25, help="Итераций Optuna"),
    min_rows: int = typer.Option(80, help="Минимум строк в датасете"),
    train_ratio: float = typer.Option(0.6, help="Доля train по времени"),
    val_ratio: float = typer.Option(0.2, help="Доля val по времени"),
    fee_bps: float = typer.Option(1.0, help="Комиссия round-trip, bps для Optuna-цели"),
    min_prob_threshold: float = typer.Option(0.35, help="Нижняя граница подбора порога вероятности"),
    max_prob_threshold: float = typer.Option(0.75, help="Верхняя граница подбора порога вероятности"),
    allow_overlap: bool = typer.Option(False, help="Разрешать перекрывающиеся сделки при оптимизации"),
    horizon: int = typer.Option(12, help="Должен совпадать с prepare-ml-dataset"),
    ret_threshold: float = typer.Option(0.0003, help="Должен совпадать с prepare-ml-dataset"),
):
    import pandas as pd

    ds = pd.read_csv(dataset_csv, parse_dates=["ts"])
    ds["ts"] = pd.to_datetime(ds["ts"], utc=True, errors="coerce")
    ds = ds.dropna(subset=["ts"])
    res = train_with_optuna(
        ds,
        out_dir,
        cfg=TrainConfig(
            n_trials=int(n_trials),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            min_rows=int(min_rows),
            fee_bps=float(fee_bps),
            min_prob_threshold=float(min_prob_threshold),
            max_prob_threshold=float(max_prob_threshold),
            allow_overlap=bool(allow_overlap),
        ),
        horizon=int(horizon),
        ret_threshold=float(ret_threshold),
    )
    print(
        f"[OK] train done best_val_score={res['best_val_score']:.4f} "
        f"best_prob_threshold={res['best_prob_threshold']:.3f} -> {res['out_dir']}"
    )
    print(f"     test_ts_start={res['test_ts_start']}")


@app.command("backtest-liquidity")
def backtest_liquidity(
    dataset_csv: str = typer.Option(..., help="Тот же CSV, что и при обучении (с close_exit)"),
    model_dir: str = typer.Option("models/liquidity_hgb", help="Каталог модели"),
    fee_bps: float = typer.Option(1.0, help="Комиссия round-trip, bps (грубо)"),
    prob_threshold: float = typer.Option(-1.0, help="Мин. вероятность выбранного класса 1 или 2; <0 => из meta.json"),
    allow_overlap: bool = typer.Option(False, help="Разрешать перекрывающиеся сделки"),
    out_equity: str = typer.Option("data/backtest_equity.csv", help="Кривая equity"),
):
    import pandas as pd

    ds = pd.read_csv(dataset_csv, parse_dates=["ts"])
    ds["ts"] = pd.to_datetime(ds["ts"], utc=True, errors="coerce")
    ds = ds.dropna(subset=["ts"])
    bt = backtest_long_short(
        ds,
        model_dir,
        fee_bps=float(fee_bps),
        prob_threshold=float(prob_threshold),
        allow_overlap=bool(allow_overlap),
    )
    Path(out_equity).parent.mkdir(parents=True, exist_ok=True)
    bt.equity_curve.to_csv(out_equity, index=False, header=["equity"])
    print(f"[OK] trades={bt.n_trades} total_return={bt.total_return:.4f} sharpe_like={bt.sharpe:.3f} max_dd={bt.max_drawdown:.4f} win_rate={bt.win_rate:.3f}")
    print(f"     equity -> {out_equity}")


@app.command("fetch-oanda")
def fetch_oanda(
    instrument: str = typer.Option(..., help="Например EUR_USD, GBP_USD, XAU_USD"),
    granularity: str = typer.Option("M1", help="S5, M1, M5, M15, M30, H1, H4, D, ..."),
    start: str = typer.Option(..., help="UTC ISO"),
    end: str = typer.Option(..., help="UTC ISO"),
    out_csv: str = typer.Option(..., help="Выход: ts,open,high,low,close,volume"),
    practice: bool = typer.Option(False, help="Practice-сервер OANDA (токен practice-аккаунта)"),
):
    """Свечи с API OANDA (брокер). Токен: переменная окружения OANDA_API_TOKEN."""
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    df = fetch_candles_range(instrument.upper(), granularity.upper(), t0, t1, cfg=OandaConfig(practice=practice))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    tag = to_pair_tag(instrument)
    print(f"[OK] OANDA {instrument} {granularity} rows={len(df)} pair_tag={tag} -> {out_csv}")


@app.command("import-histdata-csv")
def import_histdata_csv(
    path: str = typer.Option(..., help="Исходный CSV HistData"),
    out_csv: str = typer.Option(..., help="Нормализованный CSV (ts, OHLCV)"),
):
    df = load_histdata_like_csv(path)
    vr = validate_ohlcv(df)
    if not vr.ok:
        print(f"[ERROR] validation: {vr.issues}")
        raise typer.Exit(1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] rows={len(df)} -> {out_csv}")


@app.command("import-ohlcv-csv")
def import_ohlcv_csv(
    path: str = typer.Option(..., help="CSV с именованными колонками"),
    out_csv: str = typer.Option(..., help="Нормализованный CSV"),
    ts_col: str = typer.Option("ts"),
    open_col: str = typer.Option("open"),
    high_col: str = typer.Option("high"),
    low_col: str = typer.Option("low"),
    close_col: str = typer.Option("close"),
    volume_col: str = typer.Option("volume"),
    sep: str = typer.Option(","),
):
    df = load_generic_ohlcv_csv(
        path,
        ts_col=ts_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
        sep=sep,
    )
    vr = validate_ohlcv(df)
    if not vr.ok:
        print(f"[ERROR] validation: {vr.issues}")
        raise typer.Exit(1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] rows={len(df)} -> {out_csv}")


@app.command("validate-ohlcv")
def validate_ohlcv_cmd(
    path: str = typer.Option(..., help="CSV свечей"),
    histdata: bool = typer.Option(False, help="Формат HistData"),
):
    df = _load_ohlcv_file(path, histdata)
    vr = validate_ohlcv(df)
    if vr.ok:
        print(f"[OK] rows={vr.n_rows}")
    else:
        print(f"[FAIL] rows={vr.n_rows} issues={vr.issues}")
        raise typer.Exit(1)


@app.command("build-ext")
def build_ext():
    """Подсказка: pip install -e . из корня репозитория (нужен компилятор C)."""
    print("Run: pip install -e .")
    print("If build fails, the project still runs using NumPy fallbacks.")


@app.command("accounting-close")
def accounting_close(
    trades_csv: str = typer.Option(..., help="Сделки: ts,symbol,side,quantity,price[,fee]"),
    out_dir: str = typer.Option("models/accounting/close", help="Куда сохранить отчеты закрытия"),
    prices_csv: str = typer.Option("", help="Необязательные цены: symbol,price"),
    external_positions_csv: str = typer.Option("", help="Необязательная внешняя сверка: symbol,quantity"),
):
    """
    Закрытие периода по учету инвестиций:
    нормализация сделок, позиции/PnL, сверка с внешними остатками, summary.
    """
    import pandas as pd

    trades = pd.read_csv(trades_csv)
    prices = pd.read_csv(prices_csv) if prices_csv.strip() else None
    ext = pd.read_csv(external_positions_csv) if external_positions_csv.strip() else None
    summary = build_close_report(
        trades_df=trades,
        out_dir=out_dir,
        prices_df=prices,
        external_positions_df=ext,
    )
    print(
        "[OK] accounting close "
        f"trades={summary['n_trades']} symbols={summary['n_symbols']} "
        f"equity={summary['equity_estimate']:.4f} -> {out_dir}"
    )
    if summary["reconciliation"]["is_used"]:
        print(f"     reconciliation mismatches={summary['reconciliation']['n_mismatch']}")


if __name__ == "__main__":
    app()
