from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TRADE_COLS = {"ts", "symbol", "side", "quantity", "price"}


def _cash_ledger(norm: pd.DataFrame) -> pd.DataFrame:
    if norm.empty:
        return pd.DataFrame(columns=["ts", "symbol", "side", "cash_flow", "cash_balance"])
    out = norm[["ts", "symbol", "side", "cash_flow"]].copy()
    out["cash_balance"] = out["cash_flow"].cumsum()
    return out


def _kpi_rows(summary: dict) -> pd.DataFrame:
    rows = [
        {"kpi": "n_trades", "value": summary["n_trades"], "comment": "Количество учетных проводок"},
        {"kpi": "n_symbols", "value": summary["n_symbols"], "comment": "Количество инструментов"},
        {"kpi": "cash_balance", "value": summary["cash_balance"], "comment": "Итоговый денежный результат"},
        {"kpi": "market_value", "value": summary["market_value"], "comment": "Рыночная стоимость открытых позиций"},
        {"kpi": "equity_estimate", "value": summary["equity_estimate"], "comment": "Оценка собственных средств"},
        {"kpi": "realized_pnl_total", "value": summary["realized_pnl_total"], "comment": "Реализованный финансовый результат"},
        {"kpi": "unrealized_pnl_total", "value": summary["unrealized_pnl_total"], "comment": "Нереализованный финансовый результат"},
        {
            "kpi": "reconciliation_mismatches",
            "value": summary["reconciliation"]["n_mismatch"],
            "comment": "Количество расхождений при сверке",
        },
    ]
    return pd.DataFrame(rows)


def _markdown_close_report(summary: dict) -> str:
    return (
        "# Отчет закрытия периода\n\n"
        f"- Дата последней проводки: `{summary['close_ts_utc']}`\n"
        f"- Количество проводок: `{summary['n_trades']}`\n"
        f"- Количество инструментов: `{summary['n_symbols']}`\n"
        f"- Денежный результат: `{summary['cash_balance']:.6f}`\n"
        f"- Рыночная стоимость позиций: `{summary['market_value']:.6f}`\n"
        f"- Оценка собственных средств: `{summary['equity_estimate']:.6f}`\n"
        f"- Реализованный PnL: `{summary['realized_pnl_total']:.6f}`\n"
        f"- Нереализованный PnL: `{summary['unrealized_pnl_total']:.6f}`\n"
        f"- Сверка использована: `{summary['reconciliation']['is_used']}`\n"
        f"- Расхождения при сверке: `{summary['reconciliation']['n_mismatch']}`\n"
    )


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    miss = REQUIRED_TRADE_COLS - set(df.columns)
    if miss:
        raise ValueError(f"Missing required trade columns: {sorted(miss)}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["side"] = out["side"].astype(str).str.lower().str.strip()
    out = out[out["side"].isin(["buy", "sell"])].copy()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["fee"] = pd.to_numeric(out.get("fee", 0.0), errors="coerce").fillna(0.0)
    out["quantity"] = out["quantity"].abs()
    out = out.dropna(subset=["quantity", "price"])
    out = out[(out["quantity"] > 0) & (out["price"] > 0)].copy()
    out["signed_qty"] = np.where(out["side"] == "buy", out["quantity"], -out["quantity"])
    out["gross_notional"] = out["signed_qty"] * out["price"]
    # Buy reduces cash, sell increases cash.
    out["cash_flow"] = -(out["gross_notional"]) - out["fee"]
    return out.sort_values(["ts", "symbol"]).reset_index(drop=True)


def _position_pass(symbol_trades: pd.DataFrame) -> pd.DataFrame:
    pos = 0.0
    avg = 0.0
    realized_cum = 0.0
    rows: list[dict] = []
    for r in symbol_trades.itertuples(index=False):
        qty = float(r.signed_qty)
        px = float(r.price)
        old_pos = pos
        realized = 0.0
        if old_pos == 0.0 or np.sign(old_pos) == np.sign(qty):
            new_pos = old_pos + qty
            avg = ((old_pos * avg) + (qty * px)) / new_pos if abs(new_pos) > 1e-12 else 0.0
        else:
            close_qty = min(abs(old_pos), abs(qty))
            realized = close_qty * (px - avg) * np.sign(old_pos)
            new_pos = old_pos + qty
            if abs(new_pos) <= 1e-12:
                avg = 0.0
            elif np.sign(new_pos) != np.sign(old_pos):
                avg = px
        pos = new_pos
        realized_cum += realized
        rows.append(
            {
                "ts": r.ts,
                "symbol": r.symbol,
                "side": r.side,
                "price": px,
                "quantity": float(r.quantity),
                "signed_qty": qty,
                "fee": float(r.fee),
                "cash_flow": float(r.cash_flow),
                "position_qty": float(pos),
                "avg_cost": float(avg),
                "realized_pnl_trade": float(realized),
                "realized_pnl_cum": float(realized_cum),
            }
        )
    return pd.DataFrame(rows)


def _mark_map(prices_df: pd.DataFrame | None, trades: pd.DataFrame) -> dict[str, float]:
    if prices_df is not None and not prices_df.empty:
        req = {"symbol", "price"}
        miss = req - set(prices_df.columns)
        if miss:
            raise ValueError(f"Missing required price columns: {sorted(miss)}")
        px = prices_df.copy()
        px["symbol"] = px["symbol"].astype(str).str.upper().str.strip()
        px["price"] = pd.to_numeric(px["price"], errors="coerce")
        px = px.dropna(subset=["symbol", "price"])
        return dict(zip(px["symbol"], px["price"], strict=False))
    last = trades.sort_values("ts").groupby("symbol", as_index=False)["price"].last()
    return dict(zip(last["symbol"], last["price"], strict=False))


def build_positions(trades: pd.DataFrame, prices_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    norm = normalize_trades(trades)
    passes = [_position_pass(g) for _, g in norm.groupby("symbol", sort=True)]
    events = pd.concat(passes, ignore_index=True) if passes else pd.DataFrame()
    if events.empty:
        cols = ["symbol", "position_qty", "avg_cost", "mark_price", "market_value", "realized_pnl", "unrealized_pnl"]
        return events, pd.DataFrame(columns=cols)
    last = events.sort_values("ts").groupby("symbol", as_index=False).tail(1).copy()
    marks = _mark_map(prices_df, norm)
    last["mark_price"] = last["symbol"].map(marks).astype(float)
    last["market_value"] = last["position_qty"] * last["mark_price"]
    last["realized_pnl"] = last["realized_pnl_cum"]
    last["unrealized_pnl"] = (last["mark_price"] - last["avg_cost"]) * last["position_qty"]
    snapshot = last[
        ["symbol", "position_qty", "avg_cost", "mark_price", "market_value", "realized_pnl", "unrealized_pnl"]
    ].sort_values("symbol")
    return events.sort_values(["ts", "symbol"]).reset_index(drop=True), snapshot.reset_index(drop=True)


def reconcile_positions(
    internal_positions: pd.DataFrame,
    external_positions: pd.DataFrame,
    qty_tolerance: float = 1e-9,
) -> pd.DataFrame:
    ext = external_positions.copy()
    req = {"symbol", "quantity"}
    miss = req - set(ext.columns)
    if miss:
        raise ValueError(f"Missing required external columns: {sorted(miss)}")
    ext["symbol"] = ext["symbol"].astype(str).str.upper().str.strip()
    ext["quantity"] = pd.to_numeric(ext["quantity"], errors="coerce").fillna(0.0)
    left = internal_positions[["symbol", "position_qty"]].rename(columns={"position_qty": "internal_qty"})
    merged = left.merge(ext[["symbol", "quantity"]].rename(columns={"quantity": "external_qty"}), on="symbol", how="outer")
    merged["internal_qty"] = merged["internal_qty"].fillna(0.0)
    merged["external_qty"] = merged["external_qty"].fillna(0.0)
    merged["qty_diff"] = merged["internal_qty"] - merged["external_qty"]
    merged["is_match"] = merged["qty_diff"].abs() <= float(qty_tolerance)
    return merged.sort_values("symbol").reset_index(drop=True)


def build_close_report(
    trades_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    prices_df: pd.DataFrame | None = None,
    external_positions_df: pd.DataFrame | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    norm = normalize_trades(trades_df)
    events, snapshot = build_positions(norm, prices_df=prices_df)
    cash_ledger = _cash_ledger(norm)
    cash = float(norm["cash_flow"].sum()) if not norm.empty else 0.0
    market_value = float(snapshot["market_value"].sum()) if not snapshot.empty else 0.0
    realized = float(snapshot["realized_pnl"].sum()) if not snapshot.empty else 0.0
    unrealized = float(snapshot["unrealized_pnl"].sum()) if not snapshot.empty else 0.0

    norm.to_csv(out / "normalized_trades.csv", index=False)
    events.to_csv(out / "position_events.csv", index=False)
    snapshot.to_csv(out / "positions_snapshot.csv", index=False)
    cash_ledger.to_csv(out / "cash_ledger.csv", index=False)

    recon_summary = {"is_used": False, "n_mismatch": 0}
    if external_positions_df is not None:
        recon = reconcile_positions(snapshot, external_positions_df)
        recon.to_csv(out / "reconciliation_positions.csv", index=False)
        recon_summary = {
            "is_used": True,
            "n_mismatch": int((~recon["is_match"]).sum()),
        }

    summary = {
        "n_trades": int(len(norm)),
        "n_symbols": int(norm["symbol"].nunique()) if not norm.empty else 0,
        "close_ts_utc": str(norm["ts"].max()) if not norm.empty else None,
        "cash_balance": cash,
        "market_value": market_value,
        "equity_estimate": cash + market_value,
        "realized_pnl_total": realized,
        "unrealized_pnl_total": unrealized,
        "reconciliation": recon_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _kpi_rows(summary).to_csv(out / "kpi_summary.csv", index=False)
    (out / "close_report.md").write_text(_markdown_close_report(summary), encoding="utf-8")
    return summary

