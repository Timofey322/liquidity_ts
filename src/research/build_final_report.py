from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_compare(old_eq: pd.DataFrame, new_eq: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    old = old_eq.copy()
    old.columns = ["ts", "equity"]
    old["ts"] = pd.to_datetime(old["ts"], utc=True, errors="coerce")
    old = old.dropna(subset=["ts"]).sort_values("ts")

    new = new_eq.copy()
    new.columns = ["ts", "equity"]
    new["ts"] = pd.to_datetime(new["ts"], utc=True, errors="coerce")
    new = new.dropna(subset=["ts"]).sort_values("ts")

    ax.plot(old["ts"], old["equity"], label="Old MTF system (XAUUSD 1H->5M)", linewidth=1.8, color="#d62728")
    ax.plot(new["ts"], new["equity"], label="New session-POI tuned (XAUUSD)", linewidth=1.8, color="#2ca02c")
    ax.set_title("XAUUSD equity: old system vs tuned session-POI")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def main() -> None:
    root = Path(".")
    old_meta = _load_json(root / "models/compare_halfyear/xauusd_1h_5m/meta.json")
    old_eq = pd.read_csv(root / "models/compare_halfyear/xauusd_1h_5m/equity_curve.csv")

    tuned_summary = _load_json(root / "models/session_poi_xau_tuned/best_run/summary.json")
    tuned_cfg = _load_json(root / "models/session_poi_xau_tuned/best_config.json")
    tuned_eq_test = pd.read_csv(root / "models/session_poi_xau_tuned/best_run/equity_test.csv")
    tuned_eq_all = pd.read_csv(root / "models/session_poi_xau_tuned/best_run/equity_all.csv")

    baseline_summary = _load_json(root / "models/session_poi/xauusd/summary.json")
    _plot_compare(old_eq, tuned_eq_test, root / "models/final_report/charts/xau_old_vs_new_test.png")
    _plot_compare(old_eq, tuned_eq_all, root / "models/final_report/charts/xau_old_vs_new_all.png")

    old_test = {
        "total_return": float(old_meta["test_total_return"]),
        "max_drawdown": float(old_meta["test_max_drawdown"]),
        "n_trades": int(old_meta["test_n_trades"]),
        "win_rate": float(old_meta["test_win_rate"]),
        "avg_trade": float(old_meta["test_avg_trade"]),
    }
    new_test = tuned_summary["test"]
    new_all = tuned_summary["all"]

    md = f"""# FINAL REPORT

## Project showcase

This report compares:
- **Old system**: MTF ML strategy (`1H -> 5M`, `XAUUSD`) from `models/compare_halfyear`.
- **New system**: tuned **session-POI** strategy (`XAUUSD`) from `models/session_poi_xau_tuned`.

All metrics are **fee-corrected** (`fee_bps` interpreted as round-trip).

## XAUUSD: corrected old system vs tuned session-POI

| System | Scope | Total Return | Max Drawdown | Trades | Win Rate | Avg Trade |
|---|---|---:|---:|---:|---:|---:|
| Old MTF (`1H->5M`) | test | {_fmt_pct(old_test['total_return'])} | {_fmt_pct(old_test['max_drawdown'])} | {old_test['n_trades']} | {_fmt_pct(old_test['win_rate'])} | {_fmt_pct(old_test['avg_trade'])} |
| Session-POI baseline | full run | {_fmt_pct(baseline_summary['total_return'])} | {_fmt_pct(baseline_summary['max_drawdown'])} | {baseline_summary['n_trades']} | {_fmt_pct(baseline_summary['win_rate'])} | {_fmt_pct(baseline_summary['avg_trade'])} |
| Session-POI tuned | test | {_fmt_pct(new_test['total_return'])} | {_fmt_pct(new_test['max_drawdown'])} | {new_test['n_trades']} | {_fmt_pct(new_test['win_rate'])} | {_fmt_pct(new_test['avg_trade'])} |
| Session-POI tuned | full run | {_fmt_pct(new_all['total_return'])} | {_fmt_pct(new_all['max_drawdown'])} | {new_all['n_trades']} | {_fmt_pct(new_all['win_rate'])} | {_fmt_pct(new_all['avg_trade'])} |

## Key charts

- Test-equity comparison: `models/final_report/charts/xau_old_vs_new_test.png`
- Full-period comparison: `models/final_report/charts/xau_old_vs_new_all.png`
- Best session-POI trade examples: `models/session_poi_xau_tuned/best_run/charts/`

## Best tuned session-POI config (XAUUSD)

```json
{json.dumps(tuned_cfg, ensure_ascii=False, indent=2)}
```

## How the systems learn and decide

### 1) Old MTF ML system
- Learns on labeled historical bars: each bar gets target class (`up / flat / down`) over future horizon.
- Uses engineered features (returns, ATR-relative volatility, footprint proxy, fractal/FVG states, HTF context).
- Trains `HistGradientBoostingClassifier` on **train** split, selects hyperparameters/probability threshold on **validation**, then evaluates on unseen **test**.
- Entry decision: open trade when predicted probability exceeds tuned threshold and setup filters are active.
- Risk logic: stop/target rules + fee-corrected PnL in backtest.

### 2) New session-POI system (rule-based, tuned)
- Does not fit an ML classifier; instead it detects **POI zones** from session fractal sweeps and/or FVG touches.
- After POI trigger, waits for LTF confirmation (`BOS`, `CHOCH`, inversion candle; toggled in tuning).
- Trade is accepted only in selected sessions and with configured stop/target logic (nearest fractal stop + liquidity/`RR` fallback target).
- Hyperparameter tuning searches many parameter combinations and picks best by **validation score** balancing return, drawdown, and trade count.

## Portfolio-ready conclusion

- The old XAUUSD MTF ML setup remains weak out-of-sample in this research window.
- Session-POI meaningfully improves behavior on XAUUSD relative to old test baseline and keeps transparent market-structure logic.
- For portfolio publication, the tuned session-POI result should be presented as **research edge**, not guaranteed production alpha; robustness checks (additional years, alternate feeds, walk-forward re-training windows) are still recommended.
"""

    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    out_md = docs / "FINAL_REPORT.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
