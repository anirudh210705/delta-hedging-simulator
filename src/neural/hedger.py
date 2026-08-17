"""Inference-time neural hedge accounting on futures paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.hedging.engine import HedgeConfig, rebalance_schedule
from src.hedging.results import HedgeResult
from src.models.black76 import delta, payoff, price
from src.neural.features import torch_features
from src.neural.model import NeuralHedger
from src.simulation.calibration import MINUTES_PER_YEAR


@dataclass(frozen=True)
class NeuralHedgeConfig:
    hedge: HedgeConfig
    device: str = "cpu"


def _realized_volatility(
    return_sum: np.ndarray, squared_sum: np.ndarray, count: int
) -> np.ndarray:
    if count < 2:
        return np.zeros_like(return_sum)
    variance = np.maximum((squared_sum - return_sum**2 / count) / (count - 1), 0)
    return np.sqrt(variance * MINUTES_PER_YEAR)


def run_neural_hedge(
    paths: np.ndarray, model: NeuralHedger, config: NeuralHedgeConfig
) -> HedgeResult:
    """Run a causal neural residual hedge with the same accounting as baseline."""
    paths = np.asarray(paths, dtype=float)
    hedge = config.hedge
    if hedge.kind != "call" or hedge.option_position != -1:
        raise ValueError("neural prototype currently supports a short call")
    if paths.ndim != 2 or paths.shape[1] < 2 or np.any(paths <= 0):
        raise ValueError("paths must have shape (n_paths, n_steps + 1) and be positive")
    n_paths, columns = paths.shape
    n_steps = columns - 1
    schedule = rebalance_schedule(n_steps, hedge.n_rebalances)
    scheduled = np.zeros(n_steps, dtype=bool)
    scheduled[schedule] = True
    dt = hedge.maturity / n_steps
    multiplier = hedge.contract_multiplier
    premium = float(
        price(
            paths[0, 0],
            hedge.strike,
            hedge.maturity,
            hedge.rate,
            hedge.volatility,
            hedge.kind,
        )
    )
    cash = np.full(n_paths, premium * multiplier)
    position = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    trades = np.zeros(n_paths, dtype=int)
    position_ledger = np.zeros_like(paths) if hedge.keep_ledger else None
    cash_ledger = np.zeros_like(paths) if hedge.keep_ledger else None
    return_sum = np.zeros(n_paths)
    squared_sum = np.zeros(n_paths)
    model = model.to(config.device).eval()

    with torch.no_grad():
        for step in range(n_steps):
            recent = np.zeros(n_paths)
            if step > 0:
                recent = np.log(paths[:, step] / paths[:, step - 1])
                return_sum += recent
                squared_sum += recent**2
                cash *= np.exp(hedge.rate * dt)
                cash += position * (paths[:, step] - paths[:, step - 1]) * multiplier
            if scheduled[step]:
                remaining = hedge.maturity - step * dt
                baseline = np.asarray(
                    delta(
                        paths[:, step],
                        hedge.strike,
                        remaining,
                        hedge.rate,
                        hedge.volatility,
                        "call",
                    )
                )
                tensors = {
                    "current_price": torch.as_tensor(
                        paths[:, step], dtype=torch.float32, device=config.device
                    ),
                    "initial_price": torch.as_tensor(
                        paths[:, 0], dtype=torch.float32, device=config.device
                    ),
                    "black_delta": torch.as_tensor(
                        baseline, dtype=torch.float32, device=config.device
                    ),
                    "position": torch.as_tensor(
                        position, dtype=torch.float32, device=config.device
                    ),
                    "recent_return": torch.as_tensor(
                        recent, dtype=torch.float32, device=config.device
                    ),
                    "realized_volatility": torch.as_tensor(
                        _realized_volatility(return_sum, squared_sum, step),
                        dtype=torch.float32,
                        device=config.device,
                    ),
                }
                features = torch_features(
                    **tensors,
                    strike=hedge.strike,
                    remaining_maturity=remaining,
                    total_maturity=hedge.maturity,
                    pricing_volatility=hedge.volatility,
                    transaction_cost_rate=hedge.transaction_cost_rate,
                )
                adjustment = model(features).cpu().numpy()
                target = np.clip(baseline + adjustment, 0.0, 1.0)
                trade = target - position
                notional = np.abs(trade) * paths[:, step] * multiplier
                cost = hedge.transaction_cost_rate * notional
                cash -= cost
                costs += cost
                turnover += notional
                trades += np.abs(trade) > 1e-14
                position = target
            if hedge.keep_ledger:
                position_ledger[:, step] = position
                cash_ledger[:, step] = cash

    cash *= np.exp(hedge.rate * dt)
    cash += position * (paths[:, -1] - paths[:, -2]) * multiplier
    close_notional = np.abs(position) * paths[:, -1] * multiplier
    close_cost = hedge.transaction_cost_rate * close_notional
    cash -= close_cost
    costs += close_cost
    turnover += close_notional
    trades += np.abs(position) > 1e-14
    pnl = cash - payoff(paths[:, -1], hedge.strike, "call") * multiplier
    if hedge.keep_ledger:
        position_ledger[:, -1] = 0
        cash_ledger[:, -1] = cash
    return HedgeResult(
        pnl=pnl,
        transaction_costs=costs,
        turnover=turnover,
        trade_count=trades,
        initial_option_price=premium,
        rebalance_indices=schedule,
        hedge_positions=position_ledger,
        cash_accounts=cash_ledger,
    )
