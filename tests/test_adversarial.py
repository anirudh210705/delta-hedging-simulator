from dataclasses import replace

import numpy as np

from src.adversarial.search import (
    AdversarialConfig,
    controls_to_paths,
    search_adversarial_paths,
)
from src.hedging.engine import HedgeConfig


def small_config() -> AdversarialConfig:
    return AdversarialConfig(
        n_steps=30,
        n_control_points=6,
        population_size=16,
        generations=5,
        target_annualized_volatility=0.2,
        max_minute_return=0.01,
        max_total_move=0.08,
        max_terminal_move=0.02,
        top_k=4,
    )


def test_control_paths_obey_constraints() -> None:
    config = small_config()
    controls = np.random.default_rng(1).normal(size=(20, config.n_control_points))
    paths = controls_to_paths(controls, 100, config)
    returns = np.diff(np.log(paths), axis=1)
    assert np.max(np.abs(returns)) <= config.max_minute_return + 1e-12
    total_excursion = np.max(np.abs(np.log(paths / 100)))
    terminal_move = np.max(np.abs(np.log(paths[:, -1] / 100)))
    assert total_excursion <= np.log1p(config.max_total_move) + 1e-12
    assert terminal_move <= np.log1p(config.max_terminal_move) + 1e-12


def test_search_is_reproducible_and_improves() -> None:
    config = small_config()
    hedge = HedgeConfig(
        strike=100,
        maturity=1 / 252,
        volatility=0.2,
        n_rebalances=10,
        transaction_cost_rate=0.0001,
    )
    first = search_adversarial_paths(100, hedge, config, seed=7)
    second = search_adversarial_paths(100, hedge, config, seed=7)
    np.testing.assert_allclose(first.paths, second.paths)
    np.testing.assert_allclose(first.scores, second.scores)
    assert first.best_score_history[-1] >= first.best_score_history[0]
    assert first.paths.shape == (config.top_k, config.n_steps + 1)


def test_supported_objectives_produce_finite_scores() -> None:
    hedge = HedgeConfig(strike=100, n_rebalances=5, transaction_cost_rate=0.001)
    for objective in ["hedge_loss", "transaction_cost", "turnover", "near_flat_loss"]:
        result = search_adversarial_paths(
            100, hedge, replace(small_config(), objective=objective), seed=2
        )
        assert np.isfinite(result.scores).all()
