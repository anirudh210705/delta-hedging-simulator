from pathlib import Path

import numpy as np
import torch

from src.hedging.engine import HedgeConfig, run_delta_hedge
from src.neural.hedger import NeuralHedgeConfig, run_neural_hedge
from src.neural.model import FEATURE_COUNT, NeuralHedger
from src.neural.training import (
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_neural_hedger,
)
from src.simulation.gbm import GBMConfig, generate_gbm_paths


def test_model_output_shape_and_zero_initial_residual() -> None:
    model = NeuralHedger(hidden_size=8)
    features = torch.randn(5, FEATURE_COUNT)
    output = model(features)
    assert output.shape == (5,)
    torch.testing.assert_close(output, torch.zeros(5))


def test_untrained_neural_hedge_matches_black76() -> None:
    paths = generate_gbm_paths(
        100, 100, GBMConfig(volatility=0.2, horizon=1, n_steps=20), seed=2
    )
    hedge = HedgeConfig(strike=100, maturity=1, volatility=0.2, n_rebalances=10)
    baseline = run_delta_hedge(paths, hedge)
    neural = run_neural_hedge(paths, NeuralHedger(8), NeuralHedgeConfig(hedge))
    np.testing.assert_allclose(neural.pnl, baseline.pnl, atol=2e-5)


def test_neural_features_do_not_look_ahead() -> None:
    first = np.full(11, 100.0)
    second = first.copy()
    second[6:] = 120.0
    paths = np.vstack([first, second])
    hedge = HedgeConfig(
        strike=100, maturity=1, volatility=0.2, n_rebalances=10, keep_ledger=True
    )
    result = run_neural_hedge(paths, NeuralHedger(8), NeuralHedgeConfig(hedge))
    assert result.hedge_positions is not None
    np.testing.assert_allclose(
        result.hedge_positions[0, :6], result.hedge_positions[1, :6]
    )


def test_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    paths = generate_gbm_paths(
        100, 48, GBMConfig(volatility=0.2, horizon=1, n_steps=10), seed=3
    )
    model = NeuralHedger(8)
    hedge = HedgeConfig(strike=100, maturity=1, volatility=0.2, n_rebalances=5)
    config = TrainingConfig(epochs=2, batch_size=16, patience=2)
    history = train_neural_hedger(model, paths[:32], paths[32:], hedge, config)
    assert len(history) == 2
    assert np.isfinite(history[["training_loss", "validation_loss"]]).all().all()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, hedge, config)
    loaded = load_checkpoint(checkpoint)
    sample = torch.randn(4, FEATURE_COUNT)
    torch.testing.assert_close(model(sample), loaded(sample))
