# Delta Hedging & Adversarial Stock Path Generation Simulator

A quantitative-finance simulator for studying discrete option hedging under
normal, stressed, and adversarial market paths. The current foundation loads
minute-level NIFTY futures and option data, validates its quality, and provides
tested Black-76 pricing, delta, payoff, and implied-volatility calculations.

## Current scope

- Parse and validate minute-level futures, call, and put observations.
- Convert the source price units (paise) into index points.
- Decode option symbols into expiry, strike, and option type.
- Price European options on futures with Black-76.
- Calculate delta and recover implied volatility.
- Generate a data-quality summary and exploratory charts.

Path simulation, dynamic hedging, stress scenarios, and neural benchmarks will
be added in subsequent stages.

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

CSV inputs are intentionally excluded from Git. Place them in `data/` using
the schema documented in `data/README.md`.

Validate all available CSV files:

```powershell
python -m src.data.validation
```

Generate the Day 1 analysis and charts:

```powershell
python -m src.analysis.exploratory
```

Generated files are written to `outputs/day1/` and are excluded from Git.

## Quality checks

```powershell
pytest
ruff check .
```

## Data caveat

The supplied filename labels 5 February 2026 as an expiry session, but symbols
such as `NIFTY2621025750CE` decode to a 10 February 2026 expiry. Options also
retain substantial time value at the end of that session. Until the data source
confirms otherwise, the filename is treated as a label rather than proof that
the contracts expire that day.
