"""Evolutionary adversarial path search with explicit realism constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.hedging.engine import HedgeConfig, run_delta_hedge

Objective = Literal["hedge_loss", "transaction_cost", "turnover", "near_flat_loss"]
MINUTES_PER_YEAR = 252 * 375


@dataclass(frozen=True)
class AdversarialConfig:
    n_steps: int = 375
    n_control_points: int = 24
    population_size: int = 128
    generations: int = 30
    elite_fraction: float = 0.15
    mutation_scale: float = 0.35
    target_annualized_volatility: float = 0.16
    max_minute_return: float = 0.008
    max_total_move: float = 0.08
    max_terminal_move: float = 0.03
    top_k: int = 32
    objective: Objective = "hedge_loss"
    terminal_penalty: float = 2_000.0


@dataclass(frozen=True)
class AdversarialResult:
    paths: np.ndarray
    scores: np.ndarray
    best_score_history: np.ndarray
    objective: Objective

    @property
    def best_path(self) -> np.ndarray:
        return self.paths[0]


def _validate(config: AdversarialConfig) -> None:
    if config.n_steps < 2 or config.n_control_points < 2:
        raise ValueError("n_steps and n_control_points must be at least two")
    if config.population_size < 4 or config.generations < 1:
        raise ValueError("population_size must be >= 4 and generations positive")
    if not 0 < config.elite_fraction < 1:
        raise ValueError("elite_fraction must lie between zero and one")
    if config.mutation_scale <= 0 or config.target_annualized_volatility <= 0:
        raise ValueError("mutation scale and target volatility must be positive")
    if not 0 < config.max_minute_return < 1:
        raise ValueError("max_minute_return must lie between zero and one")
    if not 0 < config.max_terminal_move <= config.max_total_move < 1:
        raise ValueError("terminal and total move limits are inconsistent")
    if not 1 <= config.top_k <= config.population_size:
        raise ValueError("top_k must be between one and population_size")


def controls_to_paths(
    controls: np.ndarray, initial_price: float, config: AdversarialConfig
) -> np.ndarray:
    """Interpolate controls and enforce volatility and movement limits."""
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != config.n_control_points:
        raise ValueError("controls have the wrong shape")
    if initial_price <= 0:
        raise ValueError("initial_price must be positive")
    control_grid = np.linspace(0, config.n_steps - 1, config.n_control_points)
    step_grid = np.arange(config.n_steps)
    increments = np.vstack(
        [np.interp(step_grid, control_grid, row) for row in controls]
    )
    increments -= increments.mean(axis=1, keepdims=True)
    standard_deviation = increments.std(axis=1, keepdims=True)
    standard_deviation = np.where(standard_deviation > 1e-12, standard_deviation, 1.0)
    minute_target = config.target_annualized_volatility / np.sqrt(MINUTES_PER_YEAR)
    increments *= minute_target / standard_deviation
    increments = np.clip(
        increments, -config.max_minute_return, config.max_minute_return
    )

    max_log_move = np.log1p(config.max_total_move)
    cumulative = np.cumsum(increments, axis=1)
    excursion = np.max(np.abs(cumulative), axis=1, keepdims=True)
    scale = np.minimum(1.0, max_log_move / np.maximum(excursion, 1e-12))
    increments *= scale

    max_terminal_log = np.log1p(config.max_terminal_move)
    terminal = increments.sum(axis=1, keepdims=True)
    target_terminal = np.clip(terminal, -max_terminal_log, max_terminal_log)
    increments -= (terminal - target_terminal) / config.n_steps
    increments = np.clip(
        increments, -config.max_minute_return, config.max_minute_return
    )

    cumulative = np.cumsum(increments, axis=1)
    excursion = np.max(np.abs(cumulative), axis=1, keepdims=True)
    scale = np.minimum(1.0, max_log_move / np.maximum(excursion, 1e-12))
    cumulative *= scale
    paths = np.empty((len(controls), config.n_steps + 1))
    paths[:, 0] = initial_price
    paths[:, 1:] = initial_price * np.exp(cumulative)
    return paths


def _score(
    paths: np.ndarray, hedge_config: HedgeConfig, config: AdversarialConfig
) -> np.ndarray:
    result = run_delta_hedge(paths, hedge_config)
    if config.objective == "hedge_loss":
        return -result.pnl
    if config.objective == "transaction_cost":
        return result.transaction_costs
    if config.objective == "turnover":
        return result.turnover
    if config.objective == "near_flat_loss":
        terminal_move = np.abs(np.log(paths[:, -1] / paths[:, 0]))
        return -result.pnl - config.terminal_penalty * terminal_move
    raise ValueError(f"Unsupported objective: {config.objective!r}")


def search_adversarial_paths(
    initial_price: float,
    hedge_config: HedgeConfig,
    config: AdversarialConfig | None = None,
    *,
    seed: int | None = None,
) -> AdversarialResult:
    """Evolve smooth control points toward difficult constrained paths."""
    config = config or AdversarialConfig()
    _validate(config)
    rng = np.random.default_rng(seed)
    population = rng.standard_normal(
        (config.population_size, config.n_control_points)
    )
    elite_count = max(2, round(config.population_size * config.elite_fraction))
    history: list[float] = []
    best_controls = population.copy()
    best_scores = np.full(config.population_size, -np.inf)

    for generation in range(config.generations):
        paths = controls_to_paths(population, initial_price, config)
        scores = _score(paths, hedge_config, config)
        order = np.argsort(scores)[::-1]
        population = population[order]
        scores = scores[order]
        history.append(float(scores[0]))
        if scores[0] >= best_scores[0]:
            best_controls = population.copy()
            best_scores = scores.copy()

        elites = population[:elite_count]
        parent_indices = rng.integers(0, elite_count, config.population_size)
        decay = 1.0 - 0.75 * generation / max(config.generations - 1, 1)
        population = elites[parent_indices] + rng.normal(
            scale=config.mutation_scale * decay,
            size=(config.population_size, config.n_control_points),
        )
        population[:elite_count] = elites

    final_paths = controls_to_paths(population, initial_price, config)
    final_scores = _score(final_paths, hedge_config, config)
    combined_controls = np.vstack([best_controls, population])
    combined_scores = np.concatenate([best_scores, final_scores])
    order = np.argsort(combined_scores)[::-1][: config.top_k]
    top_paths = controls_to_paths(combined_controls[order], initial_price, config)
    return AdversarialResult(
        paths=top_paths,
        scores=combined_scores[order],
        best_score_history=np.asarray(history),
        objective=config.objective,
    )
