"""Pure helpers used by the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.metrics import HedgingMetrics, calculate_metrics
from src.hedging.engine import HedgeConfig, run_delta_hedge
from src.hedging.results import HedgeResult
from src.neural.hedger import NeuralHedgeConfig, run_neural_hedge
from src.neural.training import load_checkpoint
from src.simulation.gbm import GBMConfig, generate_gbm_paths
from src.simulation.jump_diffusion import JumpDiffusionConfig, generate_jump_paths


@dataclass(frozen=True)
class DashboardRequest:
    generator: str
    initial_price: float
    strike: float
    volatility: float
    n_paths: int
    n_steps: int
    n_rebalances: int
    transaction_cost_rate: float
    kind: str = "call"
    option_position: float = -1.0
    jump_intensity: float = 25.0
    forced_jump_size: float = 0.0
    forced_jump_step: int | None = None
    seed: int = 42
    strategy: str = "Black-76"


def simulate_dashboard(
    request: DashboardRequest,
) -> tuple[np.ndarray, HedgeResult, HedgingMetrics]:
    """Generate and hedge paths without any Streamlit dependency."""
    if request.generator == "GBM":
        paths = generate_gbm_paths(
            request.initial_price,
            request.n_paths,
            GBMConfig(
                volatility=request.volatility,
                horizon=1 / 252,
                n_steps=request.n_steps,
            ),
            seed=request.seed,
        )
    elif request.generator == "Jump diffusion":
        paths = generate_jump_paths(
            request.initial_price,
            request.n_paths,
            JumpDiffusionConfig(
                volatility=request.volatility,
                jump_intensity=request.jump_intensity,
                forced_jump_size=request.forced_jump_size,
                forced_jump_step=(
                    request.forced_jump_step
                    if request.forced_jump_size != 0
                    else None
                ),
                horizon=1 / 252,
                n_steps=request.n_steps,
            ),
            seed=request.seed,
        )
    else:
        raise ValueError(f"Unknown generator: {request.generator!r}")
    hedge = HedgeConfig(
        strike=request.strike,
        maturity=1 / 252,
        rate=0.06,
        volatility=request.volatility,
        kind=request.kind,  # type: ignore[arg-type]
        option_position=request.option_position,
        n_rebalances=request.n_rebalances,
        transaction_cost_rate=request.transaction_cost_rate,
        keep_ledger=True,
    )
    if request.strategy == "Black-76":
        result = run_delta_hedge(paths, hedge)
    elif request.strategy == "Neural residual":
        if request.kind != "call" or request.option_position != -1:
            raise ValueError("neural checkpoint supports a short call only")
        checkpoint = Path("checkpoints/neural_hedger.pt")
        if not checkpoint.exists():
            raise FileNotFoundError("Run the Day 4 benchmark to create the checkpoint")
        result = run_neural_hedge(
            paths, load_checkpoint(checkpoint), NeuralHedgeConfig(hedge)
        )
    else:
        raise ValueError(f"Unknown hedge strategy: {request.strategy!r}")
    return paths, result, calculate_metrics(result)
