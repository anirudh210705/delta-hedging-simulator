import numpy as np

from src.simulation.scenarios import default_scenarios, generate_scenario_paths


def test_all_scenarios_are_reproducible_and_positive() -> None:
    for scenario in default_scenarios(n_steps=20):
        first = generate_scenario_paths(
            scenario,
            initial_price=100,
            n_paths=4,
            volatility=0.2,
            n_steps=20,
            seed=4,
        )
        second = generate_scenario_paths(
            scenario,
            initial_price=100,
            n_paths=4,
            volatility=0.2,
            n_steps=20,
            seed=4,
        )
        assert first.shape == (4, 21)
        assert np.all(first > 0)
        np.testing.assert_array_equal(first, second)


def test_forced_scenario_shocks_land_at_midpoint() -> None:
    scenarios = {scenario.name: scenario for scenario in default_scenarios(20)}
    crash = generate_scenario_paths(
        scenarios["five_percent_crash"],
        initial_price=100,
        n_paths=2,
        volatility=0,
        n_steps=20,
        seed=1,
    )
    rally = generate_scenario_paths(
        scenarios["five_percent_rally"],
        initial_price=100,
        n_paths=2,
        volatility=0,
        n_steps=20,
        seed=1,
    )
    np.testing.assert_allclose(crash[:, 10] / crash[:, 9], 0.95)
    np.testing.assert_allclose(rally[:, 10] / rally[:, 9], 1.05)
