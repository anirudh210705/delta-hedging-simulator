"""Run stress scenarios and constrained adversarial hedge-path searches."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adversarial.search import AdversarialConfig, search_adversarial_paths
from src.data.loader import discover_market_files, futures_series, load_market_files
from src.evaluation.metrics import calculate_metrics
from src.hedging.engine import HedgeConfig, run_delta_hedge
from src.simulation.calibration import calibrate_futures
from src.simulation.scenarios import default_scenarios, generate_scenario_paths

FREQUENCIES = (5, 15, 30, 60, 100, 375)
OBJECTIVES = ("hedge_loss", "transaction_cost", "turnover", "near_flat_loss")


def _metrics_row(result: object, **labels: object) -> dict[str, object]:
    row: dict[str, object] = calculate_metrics(result).to_dict()  # type: ignore[arg-type]
    row.update(labels)
    return row


def _plot_stress_distributions(pnl: dict[str, np.ndarray], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, values in pnl.items():
        low, high = np.quantile(values, [0.01, 0.99])
        clipped = values[(values >= low) & (values <= high)]
        axis.hist(clipped, bins=60, density=True, histtype="step", label=name)
    axis.set(title="Hedge P&L under market stress", xlabel="P&L", ylabel="Density")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.savefig(output / "stress_pnl_distributions.png", dpi=160)
    plt.close(figure)


def _plot_scenario_risk(metrics: pd.DataFrame, output: Path) -> None:
    selected = metrics.loc[metrics["n_rebalances"] == 100].sort_values("cvar_95")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(selected["scenario"], selected["cvar_95"])
    axis.set(title="95% CVaR by scenario (100 hedge times)", ylabel="Loss")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "scenario_risk_comparison.png", dpi=160)
    plt.close(figure)


def _plot_adversarial_paths(
    paths: dict[str, np.ndarray], history: dict[str, np.ndarray], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for objective, path in paths.items():
        axes[0].plot(path, label=objective)
    axes[0].set(
        title="Best constrained adversarial paths",
        xlabel="Minute",
        ylabel="Futures price",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    for objective, values in history.items():
        baseline = max(abs(values[0]), 1e-12)
        relative_improvement = (values - values[0]) / baseline * 100
        axes[1].plot(relative_improvement, label=objective)
    axes[1].set(
        title="Improvement during search",
        xlabel="Generation",
        ylabel="Improvement from initial best (%)",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.savefig(output / "adversarial_paths.png", dpi=160)
    plt.close(figure)


def _plot_path_and_delta(
    path: np.ndarray, hedge_config: HedgeConfig, output: Path
) -> None:
    detailed = run_delta_hedge(path[None, :], replace(hedge_config, keep_ledger=True))
    if detailed.hedge_positions is None:
        raise RuntimeError("Expected hedge-position ledger")
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(path)
    axes[0].set(title="Worst hedge-loss path", ylabel="Futures price")
    axes[0].grid(alpha=0.2)
    axes[1].step(
        np.arange(path.size), detailed.hedge_positions[0], where="post"
    )
    axes[1].set(title="Delta hedge position", xlabel="Minute", ylabel="Contracts")
    axes[1].grid(alpha=0.2)
    figure.savefig(output / "path_and_delta.png", dpi=160)
    plt.close(figure)


def _plot_frequency_robustness(
    scenario_metrics: pd.DataFrame,
    adversarial_metrics: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    positions = np.arange(len(FREQUENCIES))
    for scenario, values in scenario_metrics.groupby("scenario"):
        values = values.set_index("n_rebalances").loc[list(FREQUENCIES)]
        axes[0].plot(
            positions, values["cvar_95"], marker="o", label=scenario
        )
    axes[0].set(title="Stress CVaR", xlabel="Hedge times", ylabel="95% CVaR")
    axes[0].set_xticks(positions, FREQUENCIES)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)
    for objective, values in adversarial_metrics.groupby("objective"):
        values = values.set_index("n_rebalances").loc[list(FREQUENCIES)]
        axes[1].plot(
            positions, values["cvar_95"], marker="o", label=objective
        )
    axes[1].set(
        title="Adversarial CVaR", xlabel="Hedge times", ylabel="95% CVaR"
    )
    axes[1].set_xticks(positions, FREQUENCIES)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)
    figure.savefig(output / "frequency_robustness.png", dpi=160)
    plt.close(figure)


def run_benchmark(
    *,
    n_paths: int = 10_000,
    seed: int = 73,
    output_dir: str | Path = "outputs/day3",
    adversarial_generations: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all normal, stress, and constrained adversarial comparisons."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    market = load_market_files(discover_market_files())
    calibration = calibrate_futures(futures_series(market))
    initial_price = calibration.initial_price
    volatility = calibration.annualized_volatility
    maturity = 1 / 252
    hedge_base = HedgeConfig(
        strike=round(initial_price / 50) * 50,
        maturity=maturity,
        rate=0.06,
        volatility=volatility,
        option_position=-1,
        n_rebalances=100,
        transaction_cost_rate=0.0001,
    )

    scenario_rows: list[dict[str, object]] = []
    scenario_pnl: dict[str, np.ndarray] = {}
    for index, scenario in enumerate(default_scenarios()):
        paths = generate_scenario_paths(
            scenario,
            initial_price=initial_price,
            n_paths=n_paths,
            volatility=volatility,
            horizon=maturity,
            seed=seed + index,
        )
        for frequency in FREQUENCIES:
            result = run_delta_hedge(
                paths, replace(hedge_base, n_rebalances=frequency)
            )
            scenario_rows.append(
                _metrics_row(
                    result, scenario=scenario.name, n_rebalances=frequency
                )
            )
            if frequency == 100:
                scenario_pnl[scenario.name] = result.pnl
    scenario_metrics = pd.DataFrame(scenario_rows)
    scenario_metrics.to_csv(output / "scenario_metrics.csv", index=False)

    adversarial_rows: list[dict[str, object]] = []
    best_paths: dict[str, np.ndarray] = {}
    histories: dict[str, np.ndarray] = {}
    for index, objective in enumerate(OBJECTIVES):
        terminal_limit = 0.005 if objective == "near_flat_loss" else 0.03
        search_config = AdversarialConfig(
            population_size=96,
            generations=adversarial_generations,
            target_annualized_volatility=max(2 * volatility, 0.12),
            max_minute_return=0.008,
            max_total_move=0.08,
            max_terminal_move=terminal_limit,
            top_k=32,
            objective=objective,  # type: ignore[arg-type]
        )
        search_hedge = replace(hedge_base, n_rebalances=60)
        adversarial = search_adversarial_paths(
            initial_price,
            search_hedge,
            search_config,
            seed=seed + 100 + index,
        )
        best_paths[objective] = adversarial.best_path
        histories[objective] = adversarial.best_score_history
        for frequency in FREQUENCIES:
            result = run_delta_hedge(
                adversarial.paths,
                replace(hedge_base, n_rebalances=frequency),
            )
            adversarial_rows.append(
                _metrics_row(
                    result,
                    objective=objective,
                    n_rebalances=frequency,
                    best_search_score=float(adversarial.scores[0]),
                )
            )
    adversarial_metrics = pd.DataFrame(adversarial_rows)
    adversarial_metrics.to_csv(output / "adversarial_metrics.csv", index=False)

    _plot_stress_distributions(scenario_pnl, output)
    _plot_scenario_risk(scenario_metrics, output)
    _plot_adversarial_paths(best_paths, histories, output)
    _plot_path_and_delta(
        best_paths["hedge_loss"], replace(hedge_base, n_rebalances=60), output
    )
    _plot_frequency_robustness(scenario_metrics, adversarial_metrics, output)
    print("Scenario results at 100 hedge times:")
    columns = ["scenario", "rmse", "cvar_95", "worst_loss", "average_turnover"]
    print(
        scenario_metrics.loc[scenario_metrics["n_rebalances"] == 100, columns]
        .sort_values("cvar_95")
        .to_string(index=False)
    )
    print(f"Wrote Day 3 benchmark to {output.resolve()}")
    return scenario_metrics, adversarial_metrics


if __name__ == "__main__":
    run_benchmark()
