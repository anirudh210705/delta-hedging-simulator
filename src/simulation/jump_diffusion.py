"""Merton jump-diffusion generator with explicit stress controls."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.simulation.common import assemble_paths, validate_path_inputs


@dataclass(frozen=True)
class JumpDiffusionConfig:
    """Annualized diffusion and Poisson jump parameters."""

    drift: float = 0.0
    volatility: float = 0.2
    jump_intensity: float = 5.0
    jump_mean: float = -0.01
    jump_volatility: float = 0.015
    volatility_multiplier: float = 1.0
    horizon: float = 1 / 252
    n_steps: int = 375
    forced_jump_step: int | None = None
    forced_jump_size: float = 0.0


STRESS_PRESETS: dict[str, dict[str, float]] = {
    "normal": {},
    "high_volatility": {"volatility_multiplier": 2.0},
    "sudden_crash": {"forced_jump_size": -0.05},
    "repeated_jumps": {"jump_intensity": 100.0},
    "volatility_spike": {"volatility_multiplier": 3.0, "jump_intensity": 25.0},
}


def stress_config(
    base: JumpDiffusionConfig, preset: str, *, crash_step: int | None = None
) -> JumpDiffusionConfig:
    """Return a config with one named stress preset applied."""
    if preset not in STRESS_PRESETS:
        raise ValueError(f"Unknown stress preset: {preset!r}")
    updates = dict(STRESS_PRESETS[preset])
    if preset == "sudden_crash":
        updates["forced_jump_step"] = crash_step or base.n_steps // 2
    return replace(base, **updates)


def generate_jump_paths(
    initial_price: float,
    n_paths: int,
    config: JumpDiffusionConfig | None = None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Generate risk-compensated Merton jump-diffusion paths."""
    config = config or JumpDiffusionConfig()
    effective_volatility = config.volatility * config.volatility_multiplier
    validate_path_inputs(
        initial_price,
        n_paths,
        config.n_steps,
        config.horizon,
        effective_volatility,
    )
    if config.jump_intensity < 0 or config.jump_volatility < 0:
        raise ValueError("jump intensity and volatility cannot be negative")
    if config.volatility_multiplier < 0:
        raise ValueError("volatility_multiplier cannot be negative")
    if config.forced_jump_size <= -1:
        raise ValueError("forced_jump_size must be greater than -1")
    if config.forced_jump_step is not None and not (
        1 <= config.forced_jump_step <= config.n_steps
    ):
        raise ValueError("forced_jump_step must be between 1 and n_steps")

    dt = config.horizon / config.n_steps
    rng = np.random.default_rng(seed)
    diffusion_shocks = rng.standard_normal((n_paths, config.n_steps))
    jump_counts = rng.poisson(config.jump_intensity * dt, (n_paths, config.n_steps))
    jump_shocks = rng.standard_normal((n_paths, config.n_steps))
    jump_logs = (
        jump_counts * config.jump_mean
        + np.sqrt(jump_counts) * config.jump_volatility * jump_shocks
    )
    expected_relative_jump = np.exp(
        config.jump_mean + 0.5 * config.jump_volatility**2
    ) - 1
    increments = (
        (
            config.drift
            - config.jump_intensity * expected_relative_jump
            - 0.5 * effective_volatility**2
        )
        * dt
        + effective_volatility * np.sqrt(dt) * diffusion_shocks
        + jump_logs
    )
    if config.forced_jump_step is not None and config.forced_jump_size != 0:
        increments[:, config.forced_jump_step - 1] += np.log1p(
            config.forced_jump_size
        )
    return assemble_paths(initial_price, increments)
