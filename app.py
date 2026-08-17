"""Interactive Streamlit interface for the delta-hedging simulator."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dashboard.service import DashboardRequest, simulate_dashboard

st.set_page_config(page_title="Delta Hedging Simulator", layout="wide")
st.title("Delta Hedging & Stress Path Simulator")
st.caption(
    "Explore how rebalancing frequency, market stress, and transaction costs "
    "change the terminal P&L of a short NIFTY call."
)


@st.cache_data(show_spinner=False)
def run_cached(request: DashboardRequest):  # type: ignore[no-untyped-def]
    return simulate_dashboard(request)


with st.sidebar:
    st.header("Simulation")
    generator = st.selectbox("Path generator", ["GBM", "Jump diffusion"])
    n_paths = st.slider("Simulated paths", 250, 5_000, 1_000, 250)
    volatility = st.slider("Annual volatility", 0.05, 0.60, 0.15, 0.01)
    initial_price = st.number_input("Initial futures price", value=25_720.0, step=50.0)
    strike = st.number_input("Option strike", value=25_700.0, step=50.0)
    option_label = st.selectbox("Option type", ["Call", "Put"])
    position_label = st.selectbox("Option position", ["Short", "Long"])
    kind = option_label.lower()
    option_position = -1.0 if position_label == "Short" else 1.0
    rebalances = st.slider("Intraday hedge times", 1, 100, 30)
    cost_bps = st.slider("Transaction cost (bps)", 0.0, 5.0, 1.0, 0.25)
    checkpoint_exists = Path("checkpoints/neural_hedger.pt").exists()
    neural_available = (
        checkpoint_exists and kind == "call" and option_position == -1
    )
    strategies = ["Black-76"] + (["Neural residual"] if neural_available else [])
    strategy = st.selectbox("Hedge strategy", strategies)
    if not checkpoint_exists:
        st.caption("Run the Day 4 benchmark to enable the neural checkpoint.")
    elif not neural_available:
        st.caption("The neural prototype currently supports a short call only.")
    jump_intensity = 25.0
    shock_size = 0.0
    shock_step = 188
    if generator == "Jump diffusion":
        jump_intensity = st.slider("Annual jump intensity", 0.0, 150.0, 25.0, 5.0)
        shock_percent = st.slider("Forced shock (%)", -10.0, 10.0, 0.0, 0.5)
        shock_size = shock_percent / 100
        shock_step = st.slider("Forced shock minute", 1, 375, 188)
    seed = st.number_input("Random seed", min_value=0, value=42)

request = DashboardRequest(
    generator=generator,
    initial_price=initial_price,
    strike=strike,
    volatility=volatility,
    n_paths=n_paths,
    n_steps=375,
    n_rebalances=rebalances,
    transaction_cost_rate=cost_bps / 10_000,
    kind=kind,
    option_position=option_position,
    jump_intensity=jump_intensity,
    forced_jump_size=shock_size,
    forced_jump_step=shock_step,
    seed=int(seed),
    strategy=strategy,
)
with st.spinner("Simulating and hedging paths..."):
    paths, result, metrics = run_cached(request)

columns = st.columns(5)
columns[0].metric("P&L RMSE", f"{metrics.rmse:,.2f}")
columns[1].metric("95% CVaR", f"{metrics.cvar_95:,.2f}")
columns[2].metric("Mean P&L", f"{metrics.mean_pnl:,.2f}")
columns[3].metric("Average cost", f"{metrics.average_transaction_cost:,.2f}")
columns[4].metric("Average turnover", f"{metrics.average_turnover:,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Sample futures paths")
    figure, axis = plt.subplots(figsize=(7, 4))
    for path in paths[: min(25, len(paths))]:
        axis.plot(path, alpha=0.35, linewidth=0.8)
    axis.set(xlabel="Minute", ylabel="Futures price")
    axis.grid(alpha=0.2)
    st.pyplot(figure)
    plt.close(figure)
with right:
    st.subheader("Terminal hedge P&L")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(result.pnl, bins=55, edgecolor="white")
    axis.axvline(0, color="black", linewidth=1)
    axis.set(xlabel="P&L", ylabel="Paths")
    axis.grid(alpha=0.2)
    st.pyplot(figure)
    plt.close(figure)

st.subheader("One path and its hedge")
figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
axes[0].plot(paths[0])
axes[0].set(ylabel="Futures price")
axes[0].grid(alpha=0.2)
if result.hedge_positions is not None:
    axes[1].step(range(paths.shape[1]), result.hedge_positions[0], where="post")
axes[1].set(xlabel="Minute", ylabel="Hedge contracts")
axes[1].grid(alpha=0.2)
st.pyplot(figure)
plt.close(figure)

with st.expander("Full risk summary"):
    st.dataframe(pd.DataFrame([metrics.to_dict()]), width="stretch")

st.caption(
    "Educational simulation only. Results depend on model assumptions and are "
    "not investment or trading advice."
)
