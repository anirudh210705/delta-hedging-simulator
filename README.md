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
- Calibrate diffusion and empirical jump parameters from futures returns.
- Generate reproducible GBM and jump-diffusion paths with stress presets.
- Run discrete Black-76 delta hedges with futures variation-margin accounting.
- Measure P&L error, VaR, CVaR, downside deviation, turnover, and costs.

Adversarial optimization and neural benchmarks will be added subsequently.

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

Run the 10,000-path Day 2 benchmark:

```powershell
python -m src.experiments.day2_benchmark
```

This writes calibrated parameters, benchmark tables, sample paths, terminal P&L
distributions, and the rebalancing/cost comparison to `outputs/day2/`.

## Hedging convention

The engine models European options on futures. Futures require no initial
notional payment; gains and losses enter the cash account through variation
margin. A positive option position means long and a negative position means
short. Terminal risk metrics define loss as negative P&L, so positive VaR and
CVaR values represent losses.

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
