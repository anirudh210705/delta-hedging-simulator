"""Named, reproducible market scenarios for stress benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.simulation.gbm import GBMConfig, generate_gbm_paths
from src.simulation.jump_diffusion import JumpDiffusionConfig, generate_jump_paths

ScenarioModel = Literal["gbm", "jump_diffusion"]


@dataclass(frozen=True)
class Scenario:
    name: str
    model: ScenarioModel
    volatility_multiplier: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = -0.01
    jump_volatility: float = 0.015
    forced_jump_step: int | None = None
    forced_jump_size: float = 0.0


def default_scenarios(n_steps: int = 375) -> tuple[Scenario, ...]:
    """Return the standard normal and stress scenario collection."""
    midpoint = n_steps // 2
    return (
        Scenario("normal_gbm", "gbm"),
        Scenario("high_volatility", "gbm", volatility_multiplier=2.0),
        Scenario(
            "five_percent_crash",
            "jump_diffusion",
            jump_intensity=0.0,
            forced_jump_step=midpoint,
            forced_jump_size=-0.05,
        ),
        Scenario(
            "five_percent_rally",
            "jump_diffusion",
            jump_intensity=0.0,
            forced_jump_step=midpoint,
            forced_jump_size=0.05,
        ),
        Scenario(
            "repeated_jumps",
            "jump_diffusion",
            jump_intensity=100.0,
            jump_mean=-0.005,
            jump_volatility=0.012,
        ),
        Scenario(
            "high_vol_jump_diffusion",
            "jump_diffusion",
            volatility_multiplier=2.5,
            jump_intensity=50.0,
            jump_mean=-0.01,
            jump_volatility=0.02,
        ),
    )


def generate_scenario_paths(
    scenario: Scenario,
    *,
    initial_price: float,
    n_paths: int,
    volatility: float,
    horizon: float = 1 / 252,
    n_steps: int = 375,
    seed: int | None = None,
) -> np.ndarray:
    """Generate paths from a named scenario using a common interface."""
    if scenario.volatility_multiplier < 0:
        raise ValueError("volatility_multiplier cannot be negative")
    stressed_volatility = volatility * scenario.volatility_multiplier
    if scenario.model == "gbm":
        return generate_gbm_paths(
            initial_price,
            n_paths,
            GBMConfig(
                drift=0.0,
                volatility=stressed_volatility,
                horizon=horizon,
                n_steps=n_steps,
            ),
            seed=seed,
        )
    if scenario.model == "jump_diffusion":
        return generate_jump_paths(
            initial_price,
            n_paths,
            JumpDiffusionConfig(
                drift=0.0,
                volatility=stressed_volatility,
                jump_intensity=scenario.jump_intensity,
                jump_mean=scenario.jump_mean,
                jump_volatility=scenario.jump_volatility,
                horizon=horizon,
                n_steps=n_steps,
                forced_jump_step=scenario.forced_jump_step,
                forced_jump_size=scenario.forced_jump_size,
            ),
            seed=seed,
        )
    raise ValueError(f"Unsupported scenario model: {scenario.model!r}")
