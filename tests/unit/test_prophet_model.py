import numpy as np
import pandas as pd
import pytest

from forecasting.prophet_model import ProphetForecaster
from forecasting.schemas import ForecastResult

RNG     = np.random.default_rng(42)
HORIZON = 30


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def series() -> pd.Series:
    """Two years of daily data: gentle trend + weekly cycle + noise."""
    n     = 365 * 2
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t     = np.arange(n)
    y     = (
        100
        + 0.05 * t                              # slow upward trend
        + 6.0 * np.sin(2 * np.pi * t / 7)      # weekly seasonality
        + RNG.normal(0, 2.0, n)
    )
    return pd.Series(y, index=dates)


@pytest.fixture(scope="module")
def forecaster(series) -> ProphetForecaster:
    """Fitted ProphetForecaster shared across all tests in this module."""
    fc = ProphetForecaster(
        yearly_seasonality=False,   # too short a history for yearly to be reliable
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    fc.fit(series)
    return fc


@pytest.fixture(scope="module")
def forecast(forecaster) -> ForecastResult:
    return forecaster.predict(HORIZON)


# ── output type ───────────────────────────────────────────────────────────────

def test_returns_forecast_result(forecast):
    assert isinstance(forecast, ForecastResult)


def test_model_name(forecast):
    assert forecast.model_name == "prophet"


# ── output shape ──────────────────────────────────────────────────────────────

def test_timestamps_length(forecast):
    assert len(forecast.timestamps) == HORIZON


def test_point_forecast_length(forecast):
    assert len(forecast.point_forecast) == HORIZON


def test_lower_80_length(forecast):
    assert len(forecast.lower_80) == HORIZON


def test_upper_80_length(forecast):
    assert len(forecast.upper_80) == HORIZON


def test_lower_95_length(forecast):
    assert len(forecast.lower_95) == HORIZON


def test_upper_95_length(forecast):
    assert len(forecast.upper_95) == HORIZON


# ── interval ordering ─────────────────────────────────────────────────────────

def test_lower_80_below_point_forecast(forecast):
    assert all(lo <= pt for lo, pt in zip(forecast.lower_80, forecast.point_forecast))


def test_upper_80_above_point_forecast(forecast):
    assert all(pt <= hi for pt, hi in zip(forecast.point_forecast, forecast.upper_80))


def test_lower_95_below_point_forecast(forecast):
    assert all(lo <= pt for lo, pt in zip(forecast.lower_95, forecast.point_forecast))


def test_upper_95_above_point_forecast(forecast):
    assert all(pt <= hi for pt, hi in zip(forecast.point_forecast, forecast.upper_95))


def test_95_lower_at_most_80_lower(forecast):
    # wider interval must have a lower (or equal) lower bound
    assert all(l95 <= l80 for l95, l80 in zip(forecast.lower_95, forecast.lower_80))


def test_95_upper_at_least_80_upper(forecast):
    # wider interval must have a higher (or equal) upper bound
    assert all(u80 <= u95 for u80, u95 in zip(forecast.upper_80, forecast.upper_95))


# ── timing ────────────────────────────────────────────────────────────────────

def test_fit_time_is_positive(forecast):
    assert forecast.fit_time_seconds > 0


# ── metadata ─────────────────────────────────────────────────────────────────

def test_metadata_contains_hyperparams(forecast):
    assert "changepoint_prior_scale" in forecast.metadata
    assert forecast.metadata["changepoint_prior_scale"] == 0.05


# ── guard rails ───────────────────────────────────────────────────────────────

def test_predict_before_fit_raises():
    fc = ProphetForecaster()
    with pytest.raises(RuntimeError, match="not been fitted"):
        fc.predict(7)


def test_integer_index_raises():
    fc = ProphetForecaster()
    with pytest.raises(ValueError, match="DatetimeIndex"):
        fc.fit(pd.Series([1.0, 2.0, 3.0]))   # default RangeIndex, not DatetimeIndex


# ── model card ────────────────────────────────────────────────────────────────

def test_card_name():
    assert ProphetForecaster().card.name == "prophet"


def test_card_handles_missing():
    assert ProphetForecaster().card.handles_missing is True


def test_card_supports_native_uncertainty():
    assert ProphetForecaster().card.supports_uncertainty == "native"


def test_card_min_samples():
    assert ProphetForecaster().card.min_samples_required > 0
