"""Result containers for hedging simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HedgeResult:
    """Pathwise terminal results and optional ledger arrays."""

    pnl: np.ndarray
    transaction_costs: np.ndarray
    turnover: np.ndarray
    trade_count: np.ndarray
    initial_option_price: float
    rebalance_indices: np.ndarray
    hedge_positions: np.ndarray | None = None
    cash_accounts: np.ndarray | None = None

    @property
    def n_paths(self) -> int:
        return len(self.pnl)
