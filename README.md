# Ferrari F80 v2

Python-проект: данные FX/металлов, аналитика и ML, **учёт сделок и закрытие периода**, Streamlit-интерфейс.

**Полное краткое описание команд и модулей:** [PROJECT.md](PROJECT.md).

## Установка

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Без компилятора C проект работает на NumPy-фолбэках (`src/fast/*_py.py`).

## Самое нужное

```bash
streamlit run tools/accounting/app.py
python -m src.cli accounting-close --trades-csv data/examples/trades.csv --out-dir models/accounting/close
pytest -q
```

Загрузка тиков/свечей, ML-пайплайн — в [PROJECT.md](PROJECT.md).
