import numpy as np
import pytest

from src.hedging.engine import HedgeConfig, rebalance_schedule, run_delta_hedge
from src.simulation.gbm import GBMConfig, generate_gbm_paths


def test_rebalance_schedule_bounds() -> None:
    np.testing.assert_array_equal(rebalance_schedule(10, 1), [0])
    assert len(rebalance_schedule(10, 10)) == 10
    with pytest.raises(ValueError, match="between"):
        rebalance_schedule(10, 11)


def test_flat_path_accounting_and_costs() -> None:
    paths = np.full((3, 11), 100.0)
    no_cost = run_delta_hedge(
        paths,
        HedgeConfig(strike=100, maturity=1, volatility=0.2, n_rebalances=10),
    )
    with_cost = run_delta_hedge(
        paths,
        HedgeConfig(
            strike=100,
            maturity=1,
            volatility=0.2,
            n_rebalances=10,
            transaction_cost_rate=0.001,
        ),
    )
    assert np.all(with_cost.transaction_costs >= 0)
    assert np.all(with_cost.pnl <= no_cost.pnl)


def test_dense_hedging_reduces_gbm_error() -> None:
    paths = generate_gbm_paths(
        100, 20_000, GBMConfig(volatility=0.2, horizon=1, n_steps=100), seed=9
    )
    sparse = run_delta_hedge(
        paths,
        HedgeConfig(strike=100, maturity=1, volatility=0.2, n_rebalances=2),
    )
    dense = run_delta_hedge(
        paths,
        HedgeConfig(strike=100, maturity=1, volatility=0.2, n_rebalances=100),
    )
    assert np.std(dense.pnl) < np.std(sparse.pnl)


def test_ledger_shape_and_invalid_paths() -> None:
    paths = np.full((2, 6), 100.0)
    result = run_delta_hedge(
        paths,
        HedgeConfig(strike=100, n_rebalances=5, keep_ledger=True),
    )
    assert result.hedge_positions is not None
    assert result.hedge_positions.shape == paths.shape
    with pytest.raises(ValueError, match="shape"):
        run_delta_hedge(np.array([100, 101]), HedgeConfig(strike=100))
