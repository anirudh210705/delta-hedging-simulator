"""Generate the Day 1 data summary and exploratory charts."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import discover_market_files, futures_series, load_market_files
from src.data.validation import validate_file

TRADING_MINUTES_PER_YEAR = 252 * 375


def generate_report(
    data_dir: str | Path = "data", output_dir: str | Path = "outputs/day1"
) -> None:
    """Write summary tables and charts for all discovered sessions."""
    files = discover_market_files(data_dir)
    frame = load_market_files(files)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    validation = pd.DataFrame(asdict(validate_file(path)) for path in files)
    validation.to_csv(output / "validation_summary.csv", index=False)

    futures = futures_series(frame)
    returns = np.log(futures).groupby(futures.index.date).diff().dropna()
    sessions = []
    for session_date, values in futures.groupby(futures.index.date):
        session_returns = np.log(values).diff().dropna()
        sessions.append(
            {
                "date": session_date,
                "open": values.iloc[0],
                "close": values.iloc[-1],
                "minimum": values.min(),
                "maximum": values.max(),
                "return_percent": 100 * (values.iloc[-1] / values.iloc[0] - 1),
                "annualized_realized_volatility": (
                    session_returns.std(ddof=1) * np.sqrt(TRADING_MINUTES_PER_YEAR)
                ),
            }
        )
    pd.DataFrame(sessions).to_csv(output / "futures_session_summary.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for session_date, values in futures.groupby(futures.index.date):
        axes[0].plot(values.index, values, label=str(session_date))
    axes[0].set(title="NIFTY futures price", ylabel="Index points")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].hist(returns * 100, bins=40, edgecolor="white")
    axes[1].set(title="One-minute log returns", xlabel="Return (%)", ylabel="Count")
    axes[1].grid(alpha=0.25)
    figure.savefig(output / "futures_and_returns.png", dpi=160)
    plt.close(figure)

    options = frame.loc[frame["instrument_type"] == "option"].copy()
    final_timestamp = options.groupby("source_file")["timestamp"].transform("max")
    closing = options.loc[final_timestamp == options["timestamp"]]
    figure, axes = plt.subplots(
        1, len(files), figsize=(12, 4), squeeze=False, constrained_layout=True
    )
    grouped_closes = closing.groupby("source_file")
    for axis, (source, values) in zip(axes[0], grouped_closes, strict=True):
        for option_type, curve in values.groupby("option_type"):
            axis.plot(curve["strike"], curve["price"], marker="o", label=option_type)
        axis.set(title=source[:8], xlabel="Strike", ylabel="Closing option price")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.savefig(output / "closing_option_curves.png", dpi=160)
    plt.close(figure)

    print(f"Wrote Day 1 report to {output.resolve()}")


if __name__ == "__main__":
    generate_report()
