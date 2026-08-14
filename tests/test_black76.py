import numpy as np
import pytest

from src.models.black76 import delta, implied_volatility, payoff, price


def test_put_call_parity() -> None:
    future, strike, maturity, rate, volatility = 100.0, 105.0, 0.5, 0.04, 0.25
    call = price(future, strike, maturity, rate, volatility, "call")
    put = price(future, strike, maturity, rate, volatility, "put")
    expected = np.exp(-rate * maturity) * (future - strike)
    assert float(call - put) == pytest.approx(expected)


def test_delta_bounds() -> None:
    call_delta = float(delta(100, 100, 1, 0.05, 0.2, "call"))
    put_delta = float(delta(100, 100, 1, 0.05, 0.2, "put"))
    assert 0 < call_delta < 1
    assert -1 < put_delta < 0


def test_price_increases_with_volatility() -> None:
    low = float(price(100, 100, 1, 0, 0.1, "call"))
    high = float(price(100, 100, 1, 0, 0.4, "call"))
    assert high > low


@pytest.mark.parametrize("kind", ["call", "put"])
def test_expiry_value_equals_payoff(kind: str) -> None:
    value = price(np.array([90, 110]), 100, 0, 0.05, 0.2, kind)
    np.testing.assert_allclose(value, payoff(np.array([90, 110]), 100, kind))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_volatility_round_trip(kind: str) -> None:
    expected = 0.32
    market = float(price(100, 105, 0.75, 0.03, expected, kind))
    actual = implied_volatility(market, 100, 105, 0.75, 0.03, kind)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_vectorized_price_is_finite() -> None:
    values = price([90, 100, 110], 100, 1, 0.03, 0.2, "call")
    assert values.shape == (3,)
    assert np.isfinite(values).all()


def test_reject_price_outside_bounds() -> None:
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_volatility(101, 100, 100, 1, 0, "call")
