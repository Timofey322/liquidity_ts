from __future__ import annotations

import json

import pandas as pd

from src.accounting.ledger import build_close_report, build_positions, normalize_trades, reconcile_positions


def test_positions_realized_unrealized_and_cashflow() -> None:
    trades = pd.DataFrame(
        [
            {"ts": "2026-01-01T10:00:00Z", "symbol": "xauusd", "side": "buy", "quantity": 10, "price": 100.0, "fee": 1.0},
            {"ts": "2026-01-01T10:10:00Z", "symbol": "xauusd", "side": "buy", "quantity": 5, "price": 110.0, "fee": 1.0},
            {"ts": "2026-01-01T10:20:00Z", "symbol": "xauusd", "side": "sell", "quantity": 8, "price": 120.0, "fee": 1.0},
        ]
    )
    norm = normalize_trades(trades)
    events, snapshot = build_positions(norm, prices_df=pd.DataFrame([{"symbol": "XAUUSD", "price": 125.0}]))

    assert len(events) == 3
    row = snapshot.iloc[0]
    assert row["symbol"] == "XAUUSD"
    assert abs(float(row["position_qty"]) - 7.0) < 1e-9
    assert abs(float(row["avg_cost"]) - 103.33333333333333) < 1e-6
    assert abs(float(row["realized_pnl"]) - 133.33333333333334) < 1e-6
    assert abs(float(row["unrealized_pnl"]) - 151.66666666666669) < 1e-6
    assert abs(float(norm["cash_flow"].sum()) + 593.0) < 1e-9


def test_reconcile_positions_detects_mismatch() -> None:
    internal = pd.DataFrame([{"symbol": "EURUSD", "position_qty": 3.0}, {"symbol": "XAUUSD", "position_qty": 7.0}])
    external = pd.DataFrame([{"symbol": "EURUSD", "quantity": 3.0}, {"symbol": "XAUUSD", "quantity": 6.0}])
    recon = reconcile_positions(internal, external)

    assert len(recon) == 2
    assert recon.loc[recon["symbol"] == "EURUSD", "is_match"].iloc[0]
    assert not recon.loc[recon["symbol"] == "XAUUSD", "is_match"].iloc[0]


def test_build_close_report_writes_standard_artifacts(tmp_path) -> None:
    trades = pd.DataFrame(
        [
            {"ts": "2026-01-01T10:00:00Z", "symbol": "XAUUSD", "side": "buy", "quantity": 1, "price": 100.0, "fee": 0.0},
            {"ts": "2026-01-01T11:00:00Z", "symbol": "XAUUSD", "side": "sell", "quantity": 1, "price": 105.0, "fee": 0.0},
        ]
    )
    summary = build_close_report(
        trades_df=trades,
        out_dir=tmp_path,
        external_positions_df=pd.DataFrame([{"symbol": "XAUUSD", "quantity": 0.0}]),
    )

    assert summary["n_trades"] == 2
    assert (tmp_path / "cash_ledger.csv").exists()
    assert (tmp_path / "kpi_summary.csv").exists()
    assert (tmp_path / "close_report.md").exists()

    parsed = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert parsed["reconciliation"]["n_mismatch"] == 0

