"""Black-76 pricing and delta for European options on futures."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

OptionKind = Literal["call", "put"]


def _validate_kind(kind: str) -> None:
    if kind not in {"call", "put"}:
        raise ValueError("kind must be 'call' or 'put'")


def payoff(futures: object, strike: object, kind: OptionKind) -> np.ndarray:
    """Return European terminal payoff."""
    _validate_kind(kind)
    futures_array = np.asarray(futures, dtype=float)
    strike_array = np.asarray(strike, dtype=float)
    sign = 1.0 if kind == "call" else -1.0
    return np.maximum(sign * (futures_array - strike_array), 0.0)


def price(
    futures: object,
    strike: object,
    maturity: object,
    rate: object,
    volatility: object,
    kind: OptionKind,
) -> np.ndarray:
    """Return the Black-76 discounted option value."""
    _validate_kind(kind)
    f, k, t, r, sigma = np.broadcast_arrays(
        *map(np.asarray, (futures, strike, maturity, rate, volatility))
    )
    f, k, t, r, sigma = [item.astype(float) for item in (f, k, t, r, sigma)]
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("futures and strike must be positive")
    if np.any(t < 0) or np.any(sigma < 0):
        raise ValueError("maturity and volatility cannot be negative")

    discount = np.exp(-r * t)
    intrinsic = discount * payoff(f, k, kind)
    active = (t > 0) & (sigma > 0)
    safe_t = np.where(active, t, 1.0)
    safe_sigma = np.where(active, sigma, 1.0)
    stddev = safe_sigma * np.sqrt(safe_t)
    d1 = (np.log(f / k) + 0.5 * stddev**2) / stddev
    d2 = d1 - stddev
    if kind == "call":
        model_value = discount * (f * ndtr(d1) - k * ndtr(d2))
    else:
        model_value = discount * (k * ndtr(-d2) - f * ndtr(-d1))
    return np.where(active, model_value, intrinsic)


def delta(
    futures: object,
    strike: object,
    maturity: object,
    rate: object,
    volatility: object,
    kind: OptionKind,
) -> np.ndarray:
    """Return Black-76 delta with respect to the futures price."""
    _validate_kind(kind)
    f, k, t, r, sigma = np.broadcast_arrays(
        *map(np.asarray, (futures, strike, maturity, rate, volatility))
    )
    f, k, t, r, sigma = [item.astype(float) for item in (f, k, t, r, sigma)]
    if np.any(f <= 0) or np.any(k <= 0):
        raise ValueError("futures and strike must be positive")
    if np.any(t < 0) or np.any(sigma < 0):
        raise ValueError("maturity and volatility cannot be negative")

    discount = np.exp(-r * t)
    sign = 1.0 if kind == "call" else -1.0
    expiry_delta = discount * np.where(sign * (f - k) > 0, sign, 0.0)
    active = (t > 0) & (sigma > 0)
    safe_t = np.where(active, t, 1.0)
    safe_sigma = np.where(active, sigma, 1.0)
    stddev = safe_sigma * np.sqrt(safe_t)
    d1 = (np.log(f / k) + 0.5 * stddev**2) / stddev
    model_delta = discount * (ndtr(d1) if kind == "call" else ndtr(d1) - 1.0)
    return np.where(active, model_delta, expiry_delta)


def implied_volatility(
    market_price: float,
    futures: float,
    strike: float,
    maturity: float,
    rate: float,
    kind: OptionKind,
    *,
    lower: float = 1e-8,
    upper: float = 5.0,
) -> float:
    """Recover volatility using a bounded Brent root solver."""
    _validate_kind(kind)
    if maturity <= 0:
        raise ValueError("maturity must be positive for implied volatility")
    intrinsic = float(np.asarray(price(futures, strike, maturity, rate, 0.0, kind)))
    upper_bound = np.exp(-rate * maturity) * (
        futures if kind == "call" else strike
    )
    tolerance = 1e-10
    if market_price < intrinsic - tolerance or market_price > upper_bound + tolerance:
        raise ValueError(
            f"market price {market_price} is outside no-arbitrage bounds "
            f"[{intrinsic}, {upper_bound}]"
        )
    if abs(market_price - intrinsic) <= tolerance:
        return 0.0

    def objective(sigma: float) -> float:
        return float(
            np.asarray(price(futures, strike, maturity, rate, sigma, kind))
        ) - market_price

    if objective(upper) < 0:
        raise ValueError(f"implied volatility exceeds upper bound {upper}")
    return float(brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12))
