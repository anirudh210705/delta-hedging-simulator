"""Loading and normalization for minute-level market data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.data.symbols import parse_symbol

REQUIRED_COLUMNS = {"date", "minute_end", "symbol", "last_trade_price"}


def load_market_data(path: str | Path, price_scale: float = 100.0) -> pd.DataFrame:
    """Load one CSV and return normalized, instrument-enriched observations."""
    path = Path(path)
    frame = pd.read_csv(path, dtype={"date": "string", "minute_end": "string"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if price_scale <= 0:
        raise ValueError("price_scale must be positive")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["date"] = frame["date"].str.zfill(8)
    frame["minute_end"] = frame["minute_end"].str.zfill(6)
    timestamp_text = frame["date"] + frame["minute_end"]
    frame["timestamp"] = pd.to_datetime(
        timestamp_text, format="%Y%m%d%H%M%S", errors="raise"
    )
    frame["raw_price"] = pd.to_numeric(frame["last_trade_price"], errors="raise")
    frame["price"] = frame["raw_price"] / price_scale

    instruments = frame["symbol"].map(parse_symbol)
    frame["instrument_type"] = instruments.map(lambda item: item.instrument_type.value)
    frame["expiry"] = pd.to_datetime(instruments.map(lambda item: item.expiry))
    frame["strike"] = instruments.map(lambda item: item.strike).astype("Float64")
    frame["option_type"] = instruments.map(
        lambda item: item.option_type.value if item.option_type else pd.NA
    ).astype("string")
    frame["source_file"] = path.name

    return frame.drop(columns=["last_trade_price"]).sort_values(
        ["timestamp", "symbol"], ignore_index=True
    )


def load_market_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate multiple market CSV files."""
    frames = [load_market_data(path) for path in paths]
    if not frames:
        raise ValueError("At least one market-data file is required")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "symbol"], ignore_index=True
    )


def discover_market_files(data_dir: str | Path = "data") -> list[Path]:
    """Return market CSV files in stable name order."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {Path(data_dir).resolve()}")
    return files


def futures_series(frame: pd.DataFrame) -> pd.Series:
    """Return the single futures price series indexed by timestamp."""
    futures = frame.loc[frame["instrument_type"] == "future", ["timestamp", "price"]]
    if futures["timestamp"].duplicated().any():
        raise ValueError("Expected one futures observation per timestamp")
    return futures.set_index("timestamp")["price"].sort_index().rename("futures_price")


def option_surface(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot option prices by timestamp, expiry, strike, and option type."""
    options = frame.loc[frame["instrument_type"] == "option"]
    return options.pivot_table(
        index="timestamp",
        columns=["expiry", "strike", "option_type"],
        values="price",
        aggfunc="first",
    ).sort_index()
