import numpy as np
import pandas as pd
import pytest

from src.simulation.calibration import calibrate_futures


def test_calibration_returns_annualized_statistics() -> None:
    index = pd.date_range("2026-02-04 09:16", periods=100, freq="min")
    prices = pd.Series(100 * np.exp(np.arange(100) * 0.0001), index=index)
    result = calibrate_futures(prices)
    assert result.initial_price == pytest.approx(prices.iloc[-1])
    assert result.observations == 100
    assert result.annualized_drift > 0
    assert result.annualized_volatility >= 0


def test_calibration_rejects_bad_prices() -> None:
    with pytest.raises(ValueError, match="positive prices"):
        calibrate_futures(pd.Series([100, 0, 101]))
