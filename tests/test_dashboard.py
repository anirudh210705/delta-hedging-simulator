import numpy as np
import pytest

from src.dashboard.service import DashboardRequest, simulate_dashboard


def test_dashboard_service_runs_end_to_end() -> None:
    request = DashboardRequest(
        generator="GBM",
        initial_price=100,
        strike=100,
        volatility=0.2,
        n_paths=20,
        n_steps=10,
        n_rebalances=5,
        transaction_cost_rate=0.0001,
    )
    paths, result, metrics = simulate_dashboard(request)
    assert paths.shape == (20, 11)
    assert result.hedge_positions is not None
    assert np.isfinite(result.pnl).all()
    assert metrics.n_paths == 20


def test_dashboard_rejects_unknown_generator() -> None:
    request = DashboardRequest(
        generator="unknown",
        initial_price=100,
        strike=100,
        volatility=0.2,
        n_paths=10,
        n_steps=10,
        n_rebalances=5,
        transaction_cost_rate=0,
    )
    with pytest.raises(ValueError, match="Unknown generator"):
        simulate_dashboard(request)


def test_dashboard_supports_long_put() -> None:
    request = DashboardRequest(
        generator="GBM",
        initial_price=100,
        strike=100,
        volatility=0.2,
        n_paths=10,
        n_steps=10,
        n_rebalances=5,
        transaction_cost_rate=0,
        kind="put",
        option_position=1,
    )
    _, result, _ = simulate_dashboard(request)
    assert np.isfinite(result.pnl).all()
