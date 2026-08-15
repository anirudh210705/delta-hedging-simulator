"""Estimate simulation inputs from observed minute futures prices."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_SESSION = 375
MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * MINUTES_PER_SESSION


@dataclass(frozen=True)
class CalibrationResult:
    initial_price: float
    observations: int
    minute_mean_log_return: float
    minute_volatility: float
    annualized_drift: float
    annualized_volatility: float
    jump_threshold: float
    jump_count: int
    annualized_jump_intensity: float
    mean_jump_log_return: float
    jump_log_return_volatility: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calibrate_futures(prices: pd.Series, jump_sigma: float = 4.0) -> CalibrationResult:
    """Calibrate annualized diffusion and empirical jump estimates."""
    clean = pd.Series(prices, dtype=float).dropna()
    if len(clean) < 3 or (clean <= 0).any():
        raise ValueError("At least three positive prices are required")
    if jump_sigma <= 0:
        raise ValueError("jump_sigma must be positive")

    if isinstance(clean.index, pd.DatetimeIndex):
        returns = np.log(clean).groupby(clean.index.date).diff().dropna()
    else:
        returns = np.log(clean).diff().dropna()
    mean = float(returns.mean())
    volatility = float(returns.std(ddof=1))
    threshold = jump_sigma * volatility
    jumps = returns.loc[(returns - mean).abs() > threshold]
    observed_years = len(returns) / MINUTES_PER_YEAR
    return CalibrationResult(
        initial_price=float(clean.iloc[-1]),
        observations=len(clean),
        minute_mean_log_return=mean,
        minute_volatility=volatility,
        annualized_drift=mean * MINUTES_PER_YEAR,
        annualized_volatility=volatility * np.sqrt(MINUTES_PER_YEAR),
        jump_threshold=threshold,
        jump_count=len(jumps),
        annualized_jump_intensity=(len(jumps) / observed_years if jumps.size else 0.0),
        mean_jump_log_return=float(jumps.mean()) if jumps.size else 0.0,
        jump_log_return_volatility=(
            float(jumps.std(ddof=1)) if jumps.size > 1 else 0.0
        ),
    )


def intraday_volatility_profile(prices: pd.Series) -> pd.Series:
    """Estimate return volatility by minute-of-session across available days."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must use a DatetimeIndex")
    returns = np.log(prices).groupby(prices.index.date).diff()
    minute_label = prices.index.strftime("%H:%M")
    return returns.groupby(minute_label).std(ddof=1).rename("minute_volatility")
