"""Parsers for the NIFTY instrument symbols used by the source data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class InstrumentType(StrEnum):
    FUTURE = "future"
    OPTION = "option"


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    instrument_type: InstrumentType
    expiry: date | None = None
    strike: float | None = None
    option_type: OptionType | None = None


_FUTURE_PATTERN = re.compile(r"^NIFTY(?P<year>\d{2})(?P<month>[A-Z]{3})FUT$")
_OPTION_PATTERN = re.compile(
    r"^NIFTY(?P<year>\d{2})(?P<month>\d)(?P<day>\d{2})"
    r"(?P<strike>\d{5})(?P<side>CE|PE)$"
)
_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_symbol(symbol: str) -> Instrument:
    """Parse a supported NIFTY futures or option symbol."""
    future_match = _FUTURE_PATTERN.fullmatch(symbol)
    if future_match:
        year = 2000 + int(future_match.group("year"))
        month = _MONTHS[future_match.group("month")]
        return Instrument(
            symbol=symbol,
            instrument_type=InstrumentType.FUTURE,
            expiry=date(year, month, 1),
        )

    option_match = _OPTION_PATTERN.fullmatch(symbol)
    if option_match:
        side = option_match.group("side")
        return Instrument(
            symbol=symbol,
            instrument_type=InstrumentType.OPTION,
            expiry=date(
                2000 + int(option_match.group("year")),
                int(option_match.group("month")),
                int(option_match.group("day")),
            ),
            strike=float(option_match.group("strike")),
            option_type=OptionType.CALL if side == "CE" else OptionType.PUT,
        )

    raise ValueError(f"Unsupported instrument symbol: {symbol!r}")
