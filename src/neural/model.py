"""Small residual policy for bounded adjustments to Black-76 delta."""

from __future__ import annotations

import torch
from torch import nn

FEATURE_COUNT = 8


class NeuralHedger(nn.Module):
    """Predict a bounded residual around a Black-76 hedge position."""

    def __init__(self, hidden_size: int = 32, max_adjustment: float = 0.25) -> None:
        super().__init__()
        if hidden_size <= 0 or max_adjustment <= 0:
            raise ValueError("hidden_size and max_adjustment must be positive")
        self.hidden_size = hidden_size
        self.max_adjustment = max_adjustment
        self.network = nn.Sequential(
            nn.Linear(FEATURE_COUNT, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the signed residual, initially exactly zero."""
        return self.max_adjustment * torch.tanh(self.network(features)).squeeze(-1)
