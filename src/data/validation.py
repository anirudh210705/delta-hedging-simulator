"""Data-quality checks and command-line report for market CSV files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data.loader import discover_market_files, load_market_data


@dataclass(frozen=True)
class ValidationReport:
    file: str
    rows: int
    trading_date: str
    first_timestamp: str
    last_timestamp: str
    timestamps: int
    symbols: int
    futures: int
    calls: int
    puts: int
    duplicate_symbol_minutes: int
    missing_values: int
    non_positive_prices: int
    complete_rectangular_panel: bool


def validate_frame(frame: pd.DataFrame, file: str = "<memory>") -> ValidationReport:
    """Validate structural properties and summarize one trading session."""
    if frame.empty:
        raise ValueError("Market data cannot be empty")
    duplicates = int(frame.duplicated(["timestamp", "symbol"]).sum())
    timestamps = frame["timestamp"].nunique()
    symbols = frame["symbol"].nunique()
    session_dates = frame["timestamp"].dt.date.unique()
    if len(session_dates) != 1:
        raise ValueError(f"Expected one trading date, found {len(session_dates)}")

    counts = frame.groupby("timestamp", observed=True)["symbol"].nunique()
    complete = bool(
        duplicates == 0
        and counts.eq(symbols).all()
        and len(frame) == timestamps * symbols
    )
    option_types = frame["option_type"].value_counts()
    return ValidationReport(
        file=file,
        rows=len(frame),
        trading_date=str(session_dates[0]),
        first_timestamp=str(frame["timestamp"].min()),
        last_timestamp=str(frame["timestamp"].max()),
        timestamps=timestamps,
        symbols=symbols,
        futures=int((frame["instrument_type"] == "future").sum() / timestamps),
        calls=int(option_types.get("call", 0) / timestamps),
        puts=int(option_types.get("put", 0) / timestamps),
        duplicate_symbol_minutes=duplicates,
        missing_values=int(frame[["timestamp", "symbol", "price"]].isna().sum().sum()),
        non_positive_prices=int((frame["price"] <= 0).sum()),
        complete_rectangular_panel=complete,
    )


def validate_file(path: str | Path) -> ValidationReport:
    """Load and validate a market CSV file."""
    path = Path(path)
    return validate_frame(load_market_data(path), file=path.name)


def main() -> None:
    reports = [validate_file(path) for path in discover_market_files()]
    print(pd.DataFrame(asdict(report) for report in reports).to_string(index=False))
    if not all(report.complete_rectangular_panel for report in reports):
        raise SystemExit("Validation failed: at least one panel is incomplete")


if __name__ == "__main__":
    main()
