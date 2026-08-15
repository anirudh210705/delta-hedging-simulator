"""Dynamic option-hedging strategies and accounting."""

from src.hedging.engine import HedgeConfig, run_delta_hedge
from src.hedging.results import HedgeResult

__all__ = ["HedgeConfig", "HedgeResult", "run_delta_hedge"]
