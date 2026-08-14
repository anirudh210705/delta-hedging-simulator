from datetime import date

import pytest

from src.data.symbols import InstrumentType, OptionType, parse_symbol


def test_parse_future() -> None:
    instrument = parse_symbol("NIFTY26FEBFUT")
    assert instrument.instrument_type is InstrumentType.FUTURE
    assert instrument.expiry == date(2026, 2, 1)
    assert instrument.strike is None


@pytest.mark.parametrize(
    ("symbol", "option_type"),
    [("NIFTY2621025750CE", OptionType.CALL), ("NIFTY2621025750PE", OptionType.PUT)],
)
def test_parse_option(symbol: str, option_type: OptionType) -> None:
    instrument = parse_symbol(symbol)
    assert instrument.instrument_type is InstrumentType.OPTION
    assert instrument.expiry == date(2026, 2, 10)
    assert instrument.strike == 25_750
    assert instrument.option_type is option_type


def test_reject_unknown_symbol() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_symbol("UNKNOWN")
