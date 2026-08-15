"""Vectorized discrete Black-76 delta hedging using futures variation margin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.hedging.results import HedgeResult
from src.models.black76 import OptionKind, delta, payoff, price


@dataclass(frozen=True)
class HedgeConfig:
    """Contract and execution assumptions for one hedge experiment."""

    strike: float
    maturity: float = 1 / 252
    rate: float = 0.0
    volatility: float = 0.2
    kind: OptionKind = "call"
    option_position: float = -1.0
    n_rebalances: int = 100
    transaction_cost_rate: float = 0.0
    contract_multiplier: float = 1.0
    keep_ledger: bool = False


def rebalance_schedule(n_steps: int, n_rebalances: int) -> np.ndarray:
    """Return evenly spaced hedge times, including inception but not expiry."""
    if not 1 <= n_rebalances <= n_steps:
        raise ValueError("n_rebalances must be between 1 and n_steps")
    return np.unique(np.linspace(0, n_steps - 1, n_rebalances, dtype=int))


def _validate(paths: np.ndarray, config: HedgeConfig) -> None:
    if paths.ndim != 2 or paths.shape[1] < 2 or paths.shape[0] < 1:
        raise ValueError("paths must have shape (n_paths, n_steps + 1)")
    if not np.isfinite(paths).all() or np.any(paths <= 0):
        raise ValueError("paths must contain positive finite prices")
    if config.strike <= 0 or config.maturity <= 0:
        raise ValueError("strike and maturity must be positive")
    if config.volatility < 0 or config.transaction_cost_rate < 0:
        raise ValueError("volatility and transaction costs cannot be negative")
    if config.contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be positive")
    if config.option_position == 0:
        raise ValueError("option_position cannot be zero")


def run_delta_hedge(paths: np.ndarray, config: HedgeConfig) -> HedgeResult:
    """Hedge an option position and return terminal P&L for every path.

    Futures have zero initial value. Their gains and losses are credited through
    variation margin as `position * price_change`. A positive option position is
    long; the hedge target is therefore the negative of option delta exposure.
    """
    paths = np.asarray(paths, dtype=float)
    _validate(paths, config)
    n_paths, columns = paths.shape
    n_steps = columns - 1
    schedule = rebalance_schedule(n_steps, config.n_rebalances)
    scheduled = np.zeros(n_steps, dtype=bool)
    scheduled[schedule] = True
    dt = config.maturity / n_steps
    multiplier = config.contract_multiplier

    initial_option_price = float(
        price(
            paths[0, 0],
            config.strike,
            config.maturity,
            config.rate,
            config.volatility,
            config.kind,
        )
    )
    cash = np.full(n_paths, -config.option_position * initial_option_price * multiplier)
    position = np.zeros(n_paths)
    total_cost = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    trade_count = np.zeros(n_paths, dtype=int)
    positions_ledger = np.zeros_like(paths) if config.keep_ledger else None
    cash_ledger = np.zeros_like(paths) if config.keep_ledger else None

    for step in range(n_steps):
        if step > 0:
            cash *= np.exp(config.rate * dt)
            cash += position * (paths[:, step] - paths[:, step - 1]) * multiplier

        if scheduled[step]:
            remaining = config.maturity - step * dt
            option_delta = delta(
                paths[:, step],
                config.strike,
                remaining,
                config.rate,
                config.volatility,
                config.kind,
            )
            target = -config.option_position * option_delta
            trade = target - position
            notional = np.abs(trade) * paths[:, step] * multiplier
            cost = config.transaction_cost_rate * notional
            cash -= cost
            total_cost += cost
            turnover += notional
            trade_count += np.abs(trade) > 1e-14
            position = target

        if config.keep_ledger:
            positions_ledger[:, step] = position
            cash_ledger[:, step] = cash

    cash *= np.exp(config.rate * dt)
    cash += position * (paths[:, -1] - paths[:, -2]) * multiplier
    close_notional = np.abs(position) * paths[:, -1] * multiplier
    close_cost = config.transaction_cost_rate * close_notional
    cash -= close_cost
    total_cost += close_cost
    turnover += close_notional
    trade_count += np.abs(position) > 1e-14
    terminal_payoff = (
        config.option_position
        * payoff(paths[:, -1], config.strike, config.kind)
        * multiplier
    )
    pnl = cash + terminal_payoff

    if config.keep_ledger:
        positions_ledger[:, -1] = 0.0
        cash_ledger[:, -1] = cash
    return HedgeResult(
        pnl=pnl,
        transaction_costs=total_cost,
        turnover=turnover,
        trade_count=trade_count,
        initial_option_price=initial_option_price,
        rebalance_indices=schedule,
        hedge_positions=positions_ledger,
        cash_accounts=cash_ledger,
    )
