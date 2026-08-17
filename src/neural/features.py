"""Causal feature construction shared by neural training and inference."""

from __future__ import annotations

import numpy as np
import torch


def torch_black76_call_delta(
    futures: torch.Tensor,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> torch.Tensor:
    """Differentiable Black-76 call delta."""
    if maturity <= 0:
        return (futures > strike).to(futures.dtype)
    stddev = max(volatility * np.sqrt(maturity), 1e-12)
    d1 = (torch.log(futures / strike) + 0.5 * stddev**2) / stddev
    normal_cdf = 0.5 * (1 + torch.erf(d1 / np.sqrt(2.0)))
    return np.exp(-rate * maturity) * normal_cdf


def torch_features(
    *,
    current_price: torch.Tensor,
    initial_price: torch.Tensor,
    strike: float,
    remaining_maturity: float,
    total_maturity: float,
    black_delta: torch.Tensor,
    position: torch.Tensor,
    recent_return: torch.Tensor,
    realized_volatility: torch.Tensor,
    pricing_volatility: float,
    transaction_cost_rate: float,
) -> torch.Tensor:
    """Build normalized features using information available at the hedge time."""
    return torch.stack(
        [
            torch.log(current_price / strike),
            torch.full_like(current_price, remaining_maturity / total_maturity),
            black_delta,
            position,
            recent_return * 100,
            realized_volatility / max(pricing_volatility, 1e-8),
            torch.full_like(current_price, transaction_cost_rate * 10_000),
            torch.log(current_price / initial_price),
        ],
        dim=1,
    )
