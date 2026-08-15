import numpy as np
import pytest

from src.simulation.gbm import GBMConfig, generate_gbm_paths
from src.simulation.jump_diffusion import (
    JumpDiffusionConfig,
    generate_jump_paths,
    stress_config,
)


def test_gbm_shape_positivity_and_seed() -> None:
    config = GBMConfig(volatility=0.2, horizon=1, n_steps=12)
    first = generate_gbm_paths(100, 50, config, seed=7)
    second = generate_gbm_paths(100, 50, config, seed=7)
    assert first.shape == (50, 13)
    assert np.all(first > 0)
    np.testing.assert_array_equal(first, second)


def test_gbm_terminal_mean_is_reasonable() -> None:
    config = GBMConfig(drift=0.05, volatility=0.2, horizon=1, n_steps=24)
    paths = generate_gbm_paths(100, 50_000, config, seed=11)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(0.05), rel=0.01)


def test_forced_crash_is_applied_at_requested_step() -> None:
    config = JumpDiffusionConfig(
        volatility=0,
        jump_intensity=0,
        n_steps=10,
        horizon=1,
        forced_jump_step=5,
        forced_jump_size=-0.05,
    )
    paths = generate_jump_paths(100, 3, config, seed=1)
    np.testing.assert_allclose(paths[:, 5] / paths[:, 4], 0.95)


def test_stress_preset_and_invalid_inputs() -> None:
    config = stress_config(JumpDiffusionConfig(n_steps=10), "high_volatility")
    assert config.volatility_multiplier == 2
    with pytest.raises(ValueError, match="initial_price"):
        generate_gbm_paths(0, 10)
    with pytest.raises(ValueError, match="Unknown"):
        stress_config(config, "unknown")
