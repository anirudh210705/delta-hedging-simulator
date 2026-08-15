import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics
from src.hedging.results import HedgeResult


def test_metrics_use_loss_as_negative_pnl() -> None:
    pnl = np.array([-4.0, -1.0, 0.0, 1.0, 2.0])
    result = HedgeResult(
        pnl=pnl,
        transaction_costs=np.ones(5),
        turnover=np.full(5, 10.0),
        trade_count=np.full(5, 2),
        initial_option_price=1.0,
        rebalance_indices=np.array([0]),
    )
    metrics = calculate_metrics(result)
    assert metrics.mean_pnl == pytest.approx(-0.4)
    assert metrics.worst_loss == 4
    assert metrics.average_transaction_cost == 1
    assert metrics.average_turnover == 10


def test_metrics_reject_empty_pnl() -> None:
    result = HedgeResult(
        pnl=np.array([]),
        transaction_costs=np.array([]),
        turnover=np.array([]),
        trade_count=np.array([]),
        initial_option_price=1.0,
        rebalance_indices=np.array([0]),
    )
    with pytest.raises(ValueError, match="non-empty"):
        calculate_metrics(result)
