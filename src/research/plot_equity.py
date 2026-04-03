from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_equity(path: str | Path) -> pd.Series:
    p = Path(path)
    df = pd.read_csv(p)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return pd.Series(df.iloc[:, 1].to_numpy(), index=pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce"), name="equity")


def main() -> None:
    out_dir = Path("models/compare_halfyear/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_configs = {
        "EURUSD": {
            "1H -> 5M": "models/compare_halfyear/eurusd_1h_5m/equity_curve.csv",
            "4H -> 15M": "models/compare_halfyear/eurusd_4h_15m/equity_curve.csv",
        },
        "GBPUSD": {
            "1H -> 5M": "models/compare_halfyear/gbpusd_1h_5m/equity_curve.csv",
            "4H -> 15M": "models/compare_halfyear/gbpusd_4h_15m/equity_curve.csv",
        },
        "XAUUSD": {
            "1H -> 5M": "models/compare_halfyear/xauusd_1h_5m/equity_curve.csv",
            "4H -> 15M": "models/compare_halfyear/xauusd_4h_15m/equity_curve.csv",
        },
    }

    tuned_summary = Path("models/xau_tuning/summary.csv")
    if tuned_summary.exists():
        tuned = pd.read_csv(tuned_summary)
        if not tuned.empty:
            best = tuned.iloc[0]["name"]
            pair_configs["XAUUSD"][f"Tuned: {best}"] = f"models/xau_tuning/{best}/equity_curve.csv"

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    for ax, (pair, curves) in zip(axes, pair_configs.items()):
        for label, path in curves.items():
            p = Path(path)
            if not p.exists():
                continue
            s = _load_equity(p)
            if s.index.inferred_type not in {"datetime64", "datetime"}:
                s.index = range(len(s))
            ax.plot(s.index, s.to_numpy(), label=label, linewidth=1.6)
        ax.set_title(f"{pair} equity curves")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Time / trade index")
    fig.suptitle("Half-year equity by pair and timeframe setup", fontsize=14)
    fig.savefig(out_dir / "equity_by_pair.png", dpi=160)
    plt.close(fig)

    for pair, curves in pair_configs.items():
        plt.figure(figsize=(12, 4.5))
        for label, path in curves.items():
            p = Path(path)
            if not p.exists():
                continue
            s = _load_equity(p)
            if s.index.inferred_type not in {"datetime64", "datetime"}:
                s.index = range(len(s))
            plt.plot(s.index, s.to_numpy(), label=label, linewidth=1.8)
        plt.title(f"{pair} equity comparison")
        plt.ylabel("Equity")
        plt.xlabel("Time / trade index")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{pair.lower()}_equity.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
