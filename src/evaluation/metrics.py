"""Risk metrics for terminal hedging profit and loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.hedging.results import HedgeResult


@dataclass(frozen=True)
class HedgingMetrics:
    n_paths: int
    mean_pnl: float
    bias: float
    pnl_std: float
    rmse: float
    median_pnl: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    worst_loss: float
    downside_deviation: float
    average_transaction_cost: float
    average_turnover: float
    average_trade_count: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _tail_risk(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    value_at_risk = float(np.quantile(losses, confidence))
    tail = losses[losses >= value_at_risk]
    return value_at_risk, float(tail.mean())


def calculate_metrics(result: HedgeResult) -> HedgingMetrics:
    """Calculate P&L metrics, defining loss as negative terminal P&L."""
    pnl = np.asarray(result.pnl, dtype=float)
    if pnl.ndim != 1 or pnl.size == 0 or not np.isfinite(pnl).all():
        raise ValueError("pnl must be a non-empty finite one-dimensional array")
    losses = -pnl
    var_95, cvar_95 = _tail_risk(losses, 0.95)
    var_99, cvar_99 = _tail_risk(losses, 0.99)
    negative_pnl = np.minimum(pnl, 0.0)
    return HedgingMetrics(
        n_paths=pnl.size,
        mean_pnl=float(pnl.mean()),
        bias=float(pnl.mean()),
        pnl_std=float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0,
        rmse=float(np.sqrt(np.mean(pnl**2))),
        median_pnl=float(np.median(pnl)),
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        worst_loss=float(losses.max()),
        downside_deviation=float(np.sqrt(np.mean(negative_pnl**2))),
        average_transaction_cost=float(np.mean(result.transaction_costs)),
        average_turnover=float(np.mean(result.turnover)),
        average_trade_count=float(np.mean(result.trade_count)),
    )
