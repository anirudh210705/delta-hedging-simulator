"""Vectorized geometric Brownian motion path generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.common import assemble_paths, validate_path_inputs


@dataclass(frozen=True)
class GBMConfig:
    """Parameters for geometric Brownian motion."""

    drift: float = 0.0
    volatility: float = 0.2
    horizon: float = 1 / 252
    n_steps: int = 375


def generate_gbm_paths(
    initial_price: float,
    n_paths: int,
    config: GBMConfig | None = None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Generate exact-discretization GBM paths with shape `(paths, steps + 1)`."""
    config = config or GBMConfig()
    validate_path_inputs(
        initial_price,
        n_paths,
        config.n_steps,
        config.horizon,
        config.volatility,
    )
    if not np.isfinite(config.drift):
        raise ValueError("drift must be finite")
    dt = config.horizon / config.n_steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, config.n_steps))
    increments = (
        (config.drift - 0.5 * config.volatility**2) * dt
        + config.volatility * np.sqrt(dt) * shocks
    )
    return assemble_paths(initial_price, increments)
