# Ferrari F80 v2 — кратко о проекте

Исследовательский репозиторий на Python: **рынок FX/металлов** (OHLCV, тики), **rule-based и ML-стратегии**, **учёт позиций и закрытие периода** с отчётами и Streamlit-дашбордом. Тяжёлые участки (фракталы, FVG, футпринт) — **Cython** (`src/fast/*.pyx`) с **NumPy-фолбэками**, если расширения не собраны.

## Структура

| Путь | Назначение |
|------|------------|
| `src/cli.py` | CLI: данные, анализ, ML-датасет, train/backtest, `accounting-close` |
| `src/accounting/` | Нормализация сделок, позиции, PnL, сверка, `close_report`, KPI |
| `src/logic/`, `src/ml/`, `src/research/` | Индикаторы, модели, weekly XGB, Session-POI и т.п. |
| `src/fast/` | Cython + фолбэки |
| `tools/accounting/` | Streamlit UI, скрипты портфеля из 3 инструментов |
| `tests/` | pytest, в т.ч. важность признаков weekly XGB |
| `models/` | Артефакты прогонов (кэши баров, equity, отчёты учёта) — локальный кэш |

## Быстрый старт

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace   # опционально; без этого — фолбэки на NumPy
pytest -q
```

## Учёт и дашборд

```bash
streamlit run tools/accounting/app.py
```

Отчёты по умолчанию: `models/accounting/streamlit_last/`.

Закрытие периода из CLI:

```bash
python -m src.cli accounting-close --trades-csv data/examples/trades.csv --out-dir models/accounting/close
```

Со сверкой и ценами:

```bash
python -m src.cli accounting-close --trades-csv data/examples/trades_real_from_strategy.csv --prices-csv data/examples/prices.csv --external-positions-csv data/examples/external_positions_zero.csv --out-dir models/accounting/close_real
```

Журнал из файла сделок стратегии → CSV для учёта:

```bash
python tools/accounting/build_accounting_trades_from_strategy.py --in-csv <trades_all.csv> --out-csv data/examples/trades_real_from_strategy.csv --symbol XAUUSD --quantity 1 --fee-per-fill 0.0
```

Портфель XAUUSD / GBPUSD / EURUSD:

```bash
python tools/accounting/run_three_instruments.py
# артефакты: models/accounting/three_instruments/
```

## Research / ML

Session-POI (rule-based контур):

```bash
python -m src.research.run_session_poi
python -m src.research.tune_session_poi_xau   # тюнинг XAUUSD
```

Weekly XGBoost:

```bash
python -m src.research.run_weekly_xgb
```

Важность признаков (пишет `models/feature_importance/weekly_xgb_last.json`):

```bash
pytest tests/test_weekly_xgb_feature_importance.py -q
```

Базовый контур ликвидности (Optuna + HGB + бэктест), пример:

```bash
python -m src.cli prepare-ml-dataset --ohlcv-csv data/eurusd_oanda.csv --pair EURUSD --out-csv data/ml_eurusd.csv --horizon 12 --ret-threshold 0.00015
python -m src.cli train-liquidity --dataset-csv data/ml_eurusd.csv --out-dir models/liquidity_hgb --n-trials 25 --horizon 12 --ret-threshold 0.00015
python -m src.cli backtest-liquidity --dataset-csv data/ml_eurusd.csv --model-dir models/liquidity_hgb
```

## Данные

В `.gitignore` исключены объёмные локальные выгрузки (скрейпы Bybit/Hyperliquid, крупные `*_duka*.csv`, `*_ml*.csv` и т.п.); в репозитории остаются **`data/examples/`** и код. Для прогонов скачайте данные командами ниже или положите CSV локально.

- **Dukascopy тики**: `fetch-duka`
- **OANDA свечи**: `fetch-oanda` (токен `OANDA_API_TOKEN`)
- **Свой OHLCV**: `import-ohlcv-csv`, `validate-ohlcv`

Пары в коде: `EURUSD`, `GBPUSD`, `XAUUSD` (в OANDA API — `EUR_USD`, `XAU_USD`).

## Диагностика ML (кратко)

Обучение weekly XGB с Optuna — порядка **нескольких минут на инструмент** при малых `n_trials`; полноценный прогон зависит от размера баров и числа trials. Имеет смысл смотреть **калибровку вероятностей** (например, пост-калибровка логистической регрессией поверх `predict_proba`) и **временной сплит** без утечки.
