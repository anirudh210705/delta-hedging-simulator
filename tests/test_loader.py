from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_market_data
from src.data.validation import validate_frame


def write_sample(path: Path) -> None:
    path.write_text(
        "date,minute_end,symbol,last_trade_price\n"
        "20260204,091600,NIFTY26FEBFUT,2575220\n"
        "20260204,091600,NIFTY2621025750CE,15300\n"
        "20260204,091700,NIFTY26FEBFUT,2575300\n"
        "20260204,091700,NIFTY2621025750CE,15400\n",
        encoding="utf-8",
    )


def test_load_normalizes_and_enriches(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    write_sample(path)
    frame = load_market_data(path)
    future = frame.loc[frame["instrument_type"] == "future"].iloc[0]
    option = frame.loc[frame["instrument_type"] == "option"].iloc[0]
    assert future["price"] == pytest.approx(25_752.2)
    assert option["price"] == pytest.approx(153.0)
    assert option["strike"] == 25_750
    assert option["option_type"] == "call"
    assert option["expiry"] == pd.Timestamp("2026-02-10")


def test_validation_detects_complete_panel(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    write_sample(path)
    report = validate_frame(load_market_data(path))
    assert report.rows == 4
    assert report.timestamps == 2
    assert report.symbols == 2
    assert report.complete_rectangular_panel


def test_missing_required_column_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("date,symbol\n20260204,NIFTY26FEBFUT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_market_data(path)
