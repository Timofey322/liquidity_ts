"""Streamlit dashboard for investment accounting."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.accounting.ledger import build_close_report
from src.accounting.strategy_trades import convert_strategy_trades_file, convert_strategy_trades_to_ledger

st.set_page_config(page_title="Investment Accounting Terminal", layout="wide")

OUT_DIR = Path("models/accounting/streamlit_last")
THREE_DIR = Path("models/accounting/three_instruments")
THREE_INPUTS = {
    "XAUUSD": Path("models/session_poi_xau_tuned/best_run/trades_all.csv"),
    "GBPUSD": Path("models/session_poi/gbpusd/trades.csv"),
    "EURUSD": Path("models/session_poi/eurusd/trades.csv"),
}
REPORT_FILES = (
    "normalized_trades.csv",
    "position_events.csv",
    "positions_snapshot.csv",
    "cash_ledger.csv",
    "reconciliation_positions.csv",
    "kpi_summary.csv",
    "summary.json",
    "close_report.md",
)


def _inject_style() -> None:
    st.markdown(
        """
<style>
.stApp {background: #0a0f14; color: #e5e7eb;}
.block-container {max-width: 1650px; padding-top: 0.6rem;}
h1,h2,h3 {color: #f8fafc; letter-spacing: .2px}
[data-testid='stMetric']{
  background: linear-gradient(180deg,#0d1520,#111827);
  border: 1px solid #243447; border-radius: 12px; padding: 9px;
}
[data-testid='stMetricLabel']{color:#94a3b8 !important}
[data-testid='stMetricValue']{color:#f8fafc !important; font-weight: 700}
[data-testid='stDataFrame']{border:1px solid #243447; border-radius: 10px}
section[data-testid='stSidebar']{background:#0c1623}
.tiny{font-size:.86rem;color:#93a4b8}
</style>
        """,
        unsafe_allow_html=True,
    )


def _zip_dir(out_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in REPORT_FILES:
            p = out_dir / name
            if p.exists():
                zf.write(p, arcname=name)
    return buf.getvalue()


def _read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return pd.read_json(path, typ="series").to_dict()


def _safe_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "final_equity" not in out.columns:
        out["final_equity"] = pd.to_numeric(out.get("equity_estimate", 0.0), errors="coerce").fillna(0.0)
    if "max_drawdown_abs" not in out.columns:
        out["max_drawdown_abs"] = 0.0
    if "realized_pnl_total" not in out.columns:
        out["realized_pnl_total"] = pd.to_numeric(out.get("cash_balance", 0.0), errors="coerce").fillna(0.0)
    if "n_trades" not in out.columns:
        out["n_trades"] = 0
    if "close_ts_utc" not in out.columns:
        out["close_ts_utc"] = ""
    if "symbol" not in out.columns:
        out["symbol"] = "UNKNOWN"
    if "n_mismatch" not in out.columns:
        if "reconciliation" in out.columns:
            extracted = (
                out["reconciliation"]
                .astype(str)
                .str.extract(r"n_mismatch'\s*:\s*(\d+)", expand=False)
                .fillna("0")
            )
            out["n_mismatch"] = pd.to_numeric(extracted, errors="coerce").fillna(0).astype(int)
        else:
            out["n_mismatch"] = 0
    out["avg_pnl_per_trade"] = np.where(
        pd.to_numeric(out["n_trades"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(out["realized_pnl_total"], errors="coerce").fillna(0.0)
        / pd.to_numeric(out["n_trades"], errors="coerce").fillna(1),
        0.0,
    )
    return out


def _equity_from_cash(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    out = df[["ts", "cash_balance"]].copy()
    out.columns = ["ts", "equity"]
    return out


def _run_single_auto_if_needed() -> tuple[bool, str]:
    if (OUT_DIR / "summary.json").exists():
        return False, "Single-run отчеты найдены."
    candidates = [
        ("data/examples/trades_real_from_strategy.csv", "data/examples/external_positions_zero.csv"),
        ("data/examples/trades.csv", "data/examples/external_positions.csv"),
    ]
    for trades_fp, ext_fp in candidates:
        p = Path(trades_fp)
        if not p.exists():
            continue
        trades = pd.read_csv(p)
        ext = pd.read_csv(ext_fp) if Path(ext_fp).exists() else None
        build_close_report(trades, OUT_DIR, external_positions_df=ext)
        return True, f"Single-run автозапуск выполнен: `{trades_fp}`"
    return False, "Single-run автозапуск пропущен (нет входного CSV)."


def _run_three() -> None:
    THREE_DIR.mkdir(parents=True, exist_ok=True)
    charts = THREE_DIR / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    per_eq: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for symbol, in_path in THREE_INPUTS.items():
        if not in_path.exists():
            raise FileNotFoundError(f"{symbol}: не найден {in_path}")
        trades = convert_strategy_trades_file(str(in_path), symbol=symbol, quantity=1.0, fee_per_fill=0.0)
        out = THREE_DIR / symbol.lower()
        s = build_close_report(trades, out, external_positions_df=pd.DataFrame([{"symbol": symbol, "quantity": 0.0}]))
        eq = _equity_from_cash(out / "cash_ledger.csv")
        per_eq[symbol] = eq
        e = eq["equity"].to_numpy(dtype=float)
        max_dd = float((e - np.maximum.accumulate(e)).min()) if e.size else 0.0
        rows.append(
            {
                "symbol": symbol,
                "n_trades": int(s["n_trades"]),
                "realized_pnl_total": float(s["realized_pnl_total"]),
                "final_equity": float(s["equity_estimate"]),
                "max_drawdown_abs": max_dd,
                "n_mismatch": int(s["reconciliation"]["n_mismatch"]),
                "close_ts_utc": s["close_ts_utc"],
            }
        )
        # Separate chart
        rebased = eq["equity"] - float(eq["equity"].iloc[0]) if not eq.empty else eq["equity"]
        fi, ai = plt.subplots(figsize=(9.8, 3.8))
        ai.plot(eq["ts"], rebased, color="#38bdf8", linewidth=1.9)
        ai.set_title(f"{symbol}: cumulative PnL")
        ai.grid(alpha=0.22)
        fi.tight_layout()
        fi.savefig(charts / f"{symbol.lower()}_equity.png", dpi=150)
        plt.close(fi)

    # Combined
    all_ts = pd.Index([])
    for df in per_eq.values():
        all_ts = all_ts.union(pd.Index(df["ts"]))
    all_ts = pd.DatetimeIndex(all_ts).sort_values()
    combined = pd.DataFrame({"ts": all_ts, "equity": 0.0})
    for df in per_eq.values():
        s = df.set_index("ts")["equity"].reindex(all_ts).ffill().fillna(0.0)
        combined["equity"] += s.to_numpy(dtype=float)

    f, ax = plt.subplots(figsize=(11.5, 5.2))
    colors = {"XAUUSD": "#f59e0b", "GBPUSD": "#22c55e", "EURUSD": "#60a5fa"}
    for symbol, df in per_eq.items():
        rebased = df["equity"] - float(df["equity"].iloc[0]) if not df.empty else df["equity"]
        ax.plot(df["ts"], rebased, label=symbol, color=colors[symbol], linewidth=1.8)
    c_rebased = combined["equity"] - float(combined["equity"].iloc[0]) if not combined.empty else combined["equity"]
    ax.plot(combined["ts"], c_rebased, label="COMBINED", color="#f8fafc", linewidth=2.6)
    ax.set_title("Portfolio cumulative PnL (rebased)")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.2)
    ax.legend()
    f.tight_layout()
    f.savefig(THREE_DIR / "equity_by_instrument_and_combined.png", dpi=160)
    plt.close(f)

    combined.to_csv(THREE_DIR / "combined_equity.csv", index=False)
    pd.DataFrame(rows).to_csv(THREE_DIR / "summary_by_instrument.csv", index=False)


def _render_single() -> None:
    st.subheader("Single-run: закрытие периода")
    summary = _read_summary(OUT_DIR / "summary.json")
    if not summary:
        st.info("Нет результатов single-run.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Проводки", int(summary["n_trades"]))
    c2.metric("Инструменты", int(summary["n_symbols"]))
    c3.metric("Equity", f"{float(summary['equity_estimate']):.4f}")
    c4.metric("Mismatch", int(summary["reconciliation"]["n_mismatch"]))
    kpi = OUT_DIR / "kpi_summary.csv"
    pos = OUT_DIR / "positions_snapshot.csv"
    cash = OUT_DIR / "cash_ledger.csv"
    t1, t2, t3, t4 = st.tabs(["KPI", "Позиции", "Cash", "JSON"])
    with t1:
        if kpi.exists():
            st.dataframe(pd.read_csv(kpi), use_container_width=True, height=320)
    with t2:
        if pos.exists():
            st.dataframe(pd.read_csv(pos), use_container_width=True, height=320)
    with t3:
        if cash.exists():
            st.dataframe(pd.read_csv(cash).tail(180), use_container_width=True, height=320)
    with t4:
        st.json(summary)
    st.download_button("Скачать отчет single-run (zip)", _zip_dir(OUT_DIR), "single_run_reports.zip", "application/zip")


def _render_portfolio() -> None:
    st.subheader("Portfolio: XAUUSD + GBPUSD + EURUSD")
    chart = THREE_DIR / "equity_by_instrument_and_combined.png"
    summary_fp = THREE_DIR / "summary_by_instrument.csv"
    combined_fp = THREE_DIR / "combined_equity.csv"
    if (not chart.exists()) or (not summary_fp.exists()) or (not combined_fp.exists()):
        _run_three()
    df = _safe_metric_cols(pd.read_csv(summary_fp))
    cdf = pd.read_csv(combined_fp)
    final_combined = float(cdf["equity"].iloc[-1]) if not cdf.empty else 0.0
    total_realized = float(df["realized_pnl_total"].sum())
    total_trades = int(df["n_trades"].sum())
    bpair = str(df.loc[df["realized_pnl_total"].idxmax(), "symbol"]) if len(df) else "-"
    wpair = str(df.loc[df["realized_pnl_total"].idxmin(), "symbol"]) if len(df) else "-"
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Инструменты", int(df["symbol"].nunique()))
    m2.metric("Проводки", total_trades)
    m3.metric("Total Realized", f"{total_realized:.4f}")
    m4.metric("Combined Equity", f"{final_combined:.4f}")
    m5.metric("Best Pair", bpair)
    m6.metric("Worst Pair", wpair)
    st.image(str(chart), caption="Cumulative PnL by instrument + combined", use_container_width=True)
    t1, t2, t3 = st.tabs(["Summary", "Combined curve", "Per instrument"])
    with t1:
        st.dataframe(
            df[["symbol", "n_trades", "realized_pnl_total", "avg_pnl_per_trade", "final_equity", "max_drawdown_abs", "n_mismatch", "close_ts_utc"]],
            use_container_width=True,
            height=300,
        )
    with t2:
        show = cdf.copy()
        show["rebased"] = show["equity"] - float(show["equity"].iloc[0]) if not show.empty else show["equity"]
        st.line_chart(show.set_index("ts")[["rebased"]], height=290)
        st.dataframe(show.tail(220), use_container_width=True, height=260)
    with t3:
        charts = THREE_DIR / "charts"
        for symbol in ["XAUUSD", "GBPUSD", "EURUSD"]:
            row = df[df["symbol"] == symbol]
            if row.empty:
                continue
            st.markdown(f"#### {symbol}")
            a, b, c, d = st.columns(4)
            a.metric("Проводки", int(row["n_trades"].iloc[0]))
            b.metric("Realized", f"{float(row['realized_pnl_total'].iloc[0]):.6f}")
            c.metric("Avg/trade", f"{float(row['avg_pnl_per_trade'].iloc[0]):.8f}")
            d.metric("MaxDD(abs)", f"{float(row['max_drawdown_abs'].iloc[0]):.6f}")
            img = charts / f"{symbol.lower()}_equity.png"
            if img.exists():
                st.image(str(img), use_container_width=True)


def _render_manual() -> None:
    st.subheader("Manual run")
    mode = st.radio(
        "Формат входа",
        ("Учетные проводки (ts, symbol, side, quantity, price[, fee])", "Журнал стратегии (entry_ts, exit_ts, side, entry, exit_price)"),
        horizontal=True,
    )
    uploaded = st.file_uploader("CSV файл сделок", type=["csv"])
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Инструмент (strategy mode)", value="XAUUSD")
    with c2:
        qty = st.number_input("Объем", min_value=0.0001, value=1.0, format="%.4f")
    with c3:
        fee = st.number_input("Комиссия за fill", min_value=0.0, value=0.0, format="%.6f")
    prices = st.file_uploader("Опционально: цены (symbol, price)", type=["csv"])
    external = st.file_uploader("Опционально: внешние позиции (symbol, quantity)", type=["csv"])
    if st.button("Запустить расчет", type="primary"):
        if uploaded is None:
            st.error("Загрузите CSV файл.")
            return
        raw = pd.read_csv(uploaded)
        if mode.startswith("Учетные"):
            trades = raw
        else:
            trades = convert_strategy_trades_to_ledger(raw, symbol=symbol.strip() or "XAUUSD", quantity=float(qty), fee_per_fill=float(fee))
        p_df = pd.read_csv(prices) if prices is not None else None
        e_df = pd.read_csv(external) if external is not None else None
        build_close_report(trades, OUT_DIR, prices_df=p_df, external_positions_df=e_df)
        st.success("Расчет завершен.")
        _render_single()


# ---------- App ----------
_inject_style()
st.title("Investment Accounting Terminal")
st.markdown("<div class='tiny'>Сценарии: Single Run • Portfolio (3 инструмента) • Manual Run</div>", unsafe_allow_html=True)
with st.sidebar:
    st.header("Управление")
    if st.button("Пересчитать портфель (3 инструмента)", use_container_width=True):
        _run_three()
        st.success("Готово")
    ran, msg = _run_single_auto_if_needed()
    st.caption(msg if msg else "—")
    st.markdown("---")
    st.caption("CLI")
    st.code("python -m src.cli accounting-close --trades-csv data/examples/trades.csv --out-dir models/accounting/close")

tab_a, tab_b, tab_c = st.tabs(["Single Run", "Portfolio 3 instruments", "Manual Run"])
with tab_a:
    _render_single()
with tab_b:
    _render_portfolio()
with tab_c:
    _render_manual()
