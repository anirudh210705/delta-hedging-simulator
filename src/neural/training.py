"""End-to-end training through differentiable futures hedge P&L."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.hedging.engine import HedgeConfig, rebalance_schedule
from src.models.black76 import price
from src.neural.features import torch_black76_call_delta, torch_features
from src.neural.model import NeuralHedger
from src.simulation.calibration import MINUTES_PER_YEAR


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    downside_weight: float = 0.25
    seed: int = 42
    device: str = "cpu"
    patience: int = 4


def _terminal_pnl(
    paths: torch.Tensor,
    model: NeuralHedger,
    hedge: HedgeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_paths, columns = paths.shape
    n_steps = columns - 1
    schedule = set(rebalance_schedule(n_steps, hedge.n_rebalances).tolist())
    dt = hedge.maturity / n_steps
    premium = float(
        price(
            float(paths[0, 0]),
            hedge.strike,
            hedge.maturity,
            hedge.rate,
            hedge.volatility,
            "call",
        )
    )
    cash = torch.full_like(paths[:, 0], premium * hedge.contract_multiplier)
    position = torch.zeros_like(cash)
    total_cost = torch.zeros_like(cash)
    return_sum = torch.zeros_like(cash)
    squared_sum = torch.zeros_like(cash)
    for step in range(n_steps):
        recent = torch.zeros_like(cash)
        if step > 0:
            recent = torch.log(paths[:, step] / paths[:, step - 1])
            return_sum = return_sum + recent
            squared_sum = squared_sum + recent.square()
            cash = cash * np.exp(hedge.rate * dt)
            cash = cash + (
                position
                * (paths[:, step] - paths[:, step - 1])
                * hedge.contract_multiplier
            )
        if step in schedule:
            remaining = hedge.maturity - step * dt
            baseline = torch_black76_call_delta(
                paths[:, step],
                hedge.strike,
                remaining,
                hedge.rate,
                hedge.volatility,
            )
            if step < 2:
                realized = torch.zeros_like(cash)
            else:
                variance = torch.clamp(
                    (squared_sum - return_sum.square() / step) / (step - 1), min=0
                )
                realized = torch.sqrt(variance * MINUTES_PER_YEAR)
            features = torch_features(
                current_price=paths[:, step],
                initial_price=paths[:, 0],
                strike=hedge.strike,
                remaining_maturity=remaining,
                total_maturity=hedge.maturity,
                black_delta=baseline,
                position=position,
                recent_return=recent,
                realized_volatility=realized,
                pricing_volatility=hedge.volatility,
                transaction_cost_rate=hedge.transaction_cost_rate,
            )
            target = torch.clamp(baseline + model(features), 0.0, 1.0)
            trade = target - position
            cost = (
                hedge.transaction_cost_rate
                * torch.abs(trade)
                * paths[:, step]
                * hedge.contract_multiplier
            )
            cash = cash - cost
            total_cost = total_cost + cost
            position = target
    cash = cash * np.exp(hedge.rate * dt)
    cash = cash + (
        position * (paths[:, -1] - paths[:, -2]) * hedge.contract_multiplier
    )
    close_cost = (
        hedge.transaction_cost_rate
        * torch.abs(position)
        * paths[:, -1]
        * hedge.contract_multiplier
    )
    total_cost = total_cost + close_cost
    cash = cash - close_cost
    payoff = torch.clamp(paths[:, -1] - hedge.strike, min=0)
    return cash - payoff * hedge.contract_multiplier, total_cost


def _loss(pnl: torch.Tensor, downside_weight: float) -> torch.Tensor:
    downside = torch.relu(-pnl)
    return pnl.square().mean() + downside_weight * downside.square().mean()


def train_neural_hedger(
    model: NeuralHedger,
    train_paths: np.ndarray,
    validation_paths: np.ndarray,
    hedge: HedgeConfig,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Train the residual policy and restore its best validation parameters."""
    config = config or TrainingConfig()
    if hedge.kind != "call" or hedge.option_position != -1:
        raise ValueError("neural prototype currently supports a short call")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    validation = torch.as_tensor(
        validation_paths, dtype=torch.float32, device=config.device
    )
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        model.train()
        indices = rng.permutation(len(train_paths))
        batch_losses = []
        for start in range(0, len(indices), config.batch_size):
            batch = torch.as_tensor(
                train_paths[indices[start : start + config.batch_size]],
                dtype=torch.float32,
                device=config.device,
            )
            optimizer.zero_grad()
            pnl, _ = _terminal_pnl(batch, model, hedge)
            loss = _loss(pnl, config.downside_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_pnl, _ = _terminal_pnl(validation, model, hedge)
            validation_loss = float(_loss(validation_pnl, config.downside_weight))
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": float(np.mean(batch_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training failed to produce a finite checkpoint")
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def save_checkpoint(
    path: str | Path,
    model: NeuralHedger,
    hedge: HedgeConfig,
    training: TrainingConfig,
) -> None:
    """Save model parameters and the configuration needed to reproduce them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_size": model.hidden_size,
            "max_adjustment": model.max_adjustment,
            "hedge_config": asdict(hedge),
            "training_config": asdict(training),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> NeuralHedger:
    """Restore a neural hedger checkpoint for deterministic inference."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = NeuralHedger(
        hidden_size=int(checkpoint["hidden_size"]),
        max_adjustment=float(checkpoint["max_adjustment"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()
