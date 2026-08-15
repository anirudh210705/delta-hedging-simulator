"""Run the reproducible Day 2 simulation and hedging benchmark."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import discover_market_files, futures_series, load_market_files
from src.evaluation.metrics import calculate_metrics
from src.hedging.engine import HedgeConfig, run_delta_hedge
from src.simulation.calibration import calibrate_futures
from src.simulation.gbm import GBMConfig, generate_gbm_paths
from src.simulation.jump_diffusion import (
    JumpDiffusionConfig,
    generate_jump_paths,
    stress_config,
)


def _save_sample_paths(
    gbm_paths: np.ndarray, stress_paths: np.ndarray, output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for path in gbm_paths[:20]:
        axes[0].plot(path, alpha=0.45, linewidth=0.8)
    axes[0].set(title="GBM sample paths", xlabel="Minute", ylabel="Futures price")
    for path in stress_paths[:20]:
        axes[1].plot(path, alpha=0.45, linewidth=0.8)
    axes[1].set(title="5% crash stress paths", xlabel="Minute")
    figure.savefig(output / "sample_paths.png", dpi=160)
    plt.close(figure)


def run_benchmark(
    *, n_paths: int = 10_000, seed: int = 42, output_dir: str | Path = "outputs/day2"
) -> pd.DataFrame:
    """Calibrate, simulate, hedge, and save the Day 2 benchmark artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    market = load_market_files(discover_market_files())
    observed_futures = futures_series(market)
    calibrated = calibrate_futures(observed_futures)
    (output / "calibrated_parameters.json").write_text(
        json.dumps(calibrated.to_dict(), indent=2), encoding="utf-8"
    )

    initial_price = calibrated.initial_price
    volatility = calibrated.annualized_volatility
    maturity = 1 / 252
    strike = round(initial_price / 50) * 50
    gbm_config = GBMConfig(
        drift=0.0, volatility=volatility, horizon=maturity, n_steps=375
    )
    paths = generate_gbm_paths(initial_price, n_paths, gbm_config, seed=seed)
    jump_base = JumpDiffusionConfig(
        drift=0.0,
        volatility=volatility,
        jump_intensity=max(calibrated.annualized_jump_intensity, 5.0),
        jump_mean=calibrated.mean_jump_log_return or -0.01,
        jump_volatility=calibrated.jump_log_return_volatility or 0.015,
        horizon=maturity,
        n_steps=375,
    )
    crash_config = stress_config(jump_base, "sudden_crash", crash_step=188)
    crash_paths = generate_jump_paths(initial_price, 20, crash_config, seed=seed + 1)
    _save_sample_paths(paths, crash_paths, output)

    base_hedge = HedgeConfig(
        strike=strike,
        maturity=maturity,
        rate=0.06,
        volatility=volatility,
        kind="call",
        option_position=-1,
    )
    rows: list[dict[str, float | int | str]] = []
    pnl_for_plot: dict[str, np.ndarray] = {}
    for cost_rate, cost_label in [(0.0, "zero_cost"), (0.0001, "one_bp")]:
        for frequency in [5, 15, 30, 60, 100, 375]:
            config = replace(
                base_hedge,
                n_rebalances=frequency,
                transaction_cost_rate=cost_rate,
            )
            result = run_delta_hedge(paths, config)
            metrics = calculate_metrics(result)
            row = metrics.to_dict()
            row.update(
                {
                    "scenario": "gbm",
                    "cost_label": cost_label,
                    "transaction_cost_rate": cost_rate,
                    "n_rebalances": frequency,
                }
            )
            rows.append(row)
            if frequency in {5, 100, 375} and cost_rate == 0:
                pnl_for_plot[f"{frequency} hedges"] = result.pnl

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(output / "benchmark_metrics.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, pnl in pnl_for_plot.items():
        axis.hist(pnl, bins=70, alpha=0.45, density=True, label=label)
    axis.set(title="Terminal hedge P&L", xlabel="P&L", ylabel="Density")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output / "pnl_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for cost_label, values in benchmark.groupby("cost_label"):
        values = values.sort_values("n_rebalances")
        axes[0].plot(
            values["n_rebalances"], values["rmse"], marker="o", label=cost_label
        )
        axes[1].plot(
            values["n_rebalances"],
            values["average_transaction_cost"],
            marker="o",
            label=cost_label,
        )
    axes[0].set(title="Hedging error", xlabel="Rebalances", ylabel="P&L RMSE")
    axes[1].set(
        title="Execution cost", xlabel="Rebalances", ylabel="Average cost"
    )
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.savefig(output / "rebalance_comparison.png", dpi=160)
    plt.close(figure)
    print(benchmark.to_string(index=False))
    print(f"Wrote Day 2 benchmark to {output.resolve()}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()
