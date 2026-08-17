"""Train and evaluate the neural hedger on held-out market regimes."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.adversarial.search import AdversarialConfig, search_adversarial_paths
from src.data.loader import discover_market_files, futures_series, load_market_files
from src.evaluation.metrics import calculate_metrics
from src.hedging.engine import HedgeConfig, run_delta_hedge
from src.neural.hedger import NeuralHedgeConfig, run_neural_hedge
from src.neural.model import NeuralHedger
from src.neural.training import TrainingConfig, save_checkpoint, train_neural_hedger
from src.simulation.calibration import calibrate_futures
from src.simulation.gbm import GBMConfig, generate_gbm_paths
from src.simulation.jump_diffusion import JumpDiffusionConfig, generate_jump_paths
from src.simulation.scenarios import Scenario, generate_scenario_paths


def _mixed_paths(
    initial_price: float,
    volatility: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    normal_count = n_paths // 2
    gbm = generate_gbm_paths(
        initial_price,
        normal_count,
        GBMConfig(volatility=volatility, horizon=1 / 252, n_steps=n_steps),
        seed=seed,
    )
    jumps = generate_jump_paths(
        initial_price,
        n_paths - normal_count,
        JumpDiffusionConfig(
            volatility=volatility * 1.5,
            jump_intensity=50,
            jump_mean=-0.005,
            jump_volatility=0.01,
            horizon=1 / 252,
            n_steps=n_steps,
        ),
        seed=seed + 1,
    )
    combined = np.vstack([gbm, jumps])
    return combined[np.random.default_rng(seed).permutation(len(combined))]


def _save_plots(
    history: pd.DataFrame,
    benchmark: pd.DataFrame,
    pnl: dict[tuple[str, str], np.ndarray],
    output: Path,
) -> None:
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["training_loss"], label="training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set(title="Neural hedge training", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    normal = benchmark.loc[benchmark["scenario"] == "normal_gbm"]
    axes[1].bar(normal["strategy"], normal["rmse"])
    axes[1].set(title="Held-out normal-path error", ylabel="P&L RMSE")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output / "training_and_benchmark.png", dpi=160)
    figure.savefig(assets / "neural_benchmark.png", dpi=160)
    plt.close(figure)

    scenarios = sorted({key[1] for key in pnl})
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(4 * len(scenarios), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, scenario in zip(axes, scenarios, strict=True):
        for strategy in ["black76", "neural"]:
            values = pnl[(strategy, scenario)]
            low, high = np.quantile(values, [0.01, 0.99])
            values = values[(values >= low) & (values <= high)]
            axis.hist(values, bins=45, density=True, alpha=0.45, label=strategy)
        axis.set(title=scenario, xlabel="P&L")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    figure.savefig(output / "strategy_pnl_comparison.png", dpi=160)
    figure.savefig(assets / "strategy_pnl_comparison.png", dpi=160)
    plt.close(figure)


def run_benchmark(
    *,
    train_size: int = 6_000,
    validation_size: int = 1_500,
    test_size: int = 10_000,
    n_steps: int = 100,
    seed: int = 101,
    output_dir: str | Path = "outputs/day4",
) -> pd.DataFrame:
    """Train once and compare baseline and neural hedges on unseen paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibration = calibrate_futures(
        futures_series(load_market_files(discover_market_files()))
    )
    initial_price = calibration.initial_price
    volatility = calibration.annualized_volatility
    hedge = HedgeConfig(
        strike=round(initial_price / 50) * 50,
        maturity=1 / 252,
        rate=0.06,
        volatility=volatility,
        kind="call",
        option_position=-1,
        n_rebalances=30,
        transaction_cost_rate=0.0001,
    )
    train_paths = _mixed_paths(initial_price, volatility, train_size, n_steps, seed)
    validation_paths = _mixed_paths(
        initial_price, volatility, validation_size, n_steps, seed + 10
    )
    torch.manual_seed(seed)
    model = NeuralHedger(hidden_size=32, max_adjustment=0.2)
    training_config = TrainingConfig(
        epochs=10, batch_size=512, learning_rate=8e-4, seed=seed, patience=4
    )
    history = train_neural_hedger(
        model, train_paths, validation_paths, hedge, training_config
    )
    history.to_csv(output / "training_history.csv", index=False)
    save_checkpoint("checkpoints/neural_hedger.pt", model, hedge, training_config)

    scenarios = {
        "normal_gbm": generate_gbm_paths(
            initial_price,
            test_size,
            GBMConfig(volatility=volatility, horizon=1 / 252, n_steps=n_steps),
            seed=seed + 20,
        ),
        "jump_diffusion": generate_jump_paths(
            initial_price,
            test_size,
            JumpDiffusionConfig(
                volatility=volatility * 1.5,
                jump_intensity=50,
                jump_mean=-0.005,
                jump_volatility=0.01,
                horizon=1 / 252,
                n_steps=n_steps,
            ),
            seed=seed + 21,
        ),
        "five_percent_crash": generate_scenario_paths(
            Scenario(
                "five_percent_crash",
                "jump_diffusion",
                forced_jump_step=n_steps // 2,
                forced_jump_size=-0.05,
            ),
            initial_price=initial_price,
            n_paths=test_size,
            volatility=volatility,
            horizon=1 / 252,
            n_steps=n_steps,
            seed=seed + 22,
        ),
    }
    search = search_adversarial_paths(
        initial_price,
        hedge,
        AdversarialConfig(
            n_steps=n_steps,
            n_control_points=16,
            population_size=128,
            generations=15,
            target_annualized_volatility=max(2 * volatility, 0.12),
            top_k=64,
        ),
        seed=seed + 23,
    )
    scenarios["adversarial"] = search.paths

    rows: list[dict[str, object]] = []
    pnl: dict[tuple[str, str], np.ndarray] = {}
    for scenario, paths in scenarios.items():
        results = {
            "black76": run_delta_hedge(paths, hedge),
            "neural": run_neural_hedge(paths, model, NeuralHedgeConfig(hedge)),
        }
        for strategy, result in results.items():
            row: dict[str, object] = calculate_metrics(result).to_dict()
            row.update(strategy=strategy, scenario=scenario)
            rows.append(row)
            pnl[(strategy, scenario)] = result.pnl
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(output / "neural_benchmark.csv", index=False)
    _save_plots(history, benchmark, pnl, output)
    columns = [
        "scenario",
        "strategy",
        "rmse",
        "cvar_95",
        "average_transaction_cost",
        "average_turnover",
    ]
    print(benchmark[columns].to_string(index=False))
    print(f"Wrote Day 4 benchmark to {output.resolve()}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()
