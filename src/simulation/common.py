"""Shared validation for vectorized path generators."""

from __future__ import annotations

import numpy as np


def validate_path_inputs(
    initial_price: float,
    n_paths: int,
    n_steps: int,
    horizon: float,
    volatility: float,
) -> None:
    """Validate the common simulation parameters."""
    if not np.isfinite(initial_price) or initial_price <= 0:
        raise ValueError("initial_price must be positive and finite")
    if not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and finite")
    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be non-negative and finite")


def assemble_paths(initial_price: float, log_increments: np.ndarray) -> np.ndarray:
    """Convert simulated log increments into positive price paths."""
    cumulative = np.cumsum(log_increments, axis=1)
    paths = np.empty((log_increments.shape[0], log_increments.shape[1] + 1))
    paths[:, 0] = initial_price
    paths[:, 1:] = initial_price * np.exp(cumulative)
    if not np.isfinite(paths).all() or np.any(paths <= 0):
        raise FloatingPointError("simulation produced invalid prices")
    return paths
