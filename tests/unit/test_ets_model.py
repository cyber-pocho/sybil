import numpy as np
import pandas as pd
import pytest

from forecasting.ets_model import ETSForecaster
from forecasting.schemas import ForecastResult

RNG     = np.random.default_rng(7)
HORIZON = 14


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def series() -> pd.Series:
    """Two years of daily data: linear trend + weekly cycle + low noise."""
    n     = 365 * 2
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t     = np.arange(n)
    y     = (
        80
        + 0.04 * t                              # gentle upward trend
        + 5.0 * np.sin(2 * np.pi * t / 7)      # weekly seasonality
        + RNG.normal(0, 1.5, n)
    )
    return pd.Series(y, index=dates)


@pytest.fixture(scope="module")
def forecaster(series) -> ETSForecaster:
    """Fitted ETSForecaster shared across all tests in this module."""
    fc = ETSForecaster(
        seasonal_periods=7,
        error="add",
        trend="add",
        seasonal="add",
        damped_trend=True,
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
    assert forecast.model_name == "ets"


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


def test_95_interval_at_least_as_wide_as_80(forecast):
    # 95 % lower ≤ 80 % lower and 95 % upper ≥ 80 % upper
    assert all(l95 <= l80 for l95, l80 in zip(forecast.lower_95, forecast.lower_80))
    assert all(u95 >= u80 for u95, u80 in zip(forecast.upper_95, forecast.upper_80))


# ── timestamps ────────────────────────────────────────────────────────────────

def test_timestamps_are_after_training_end(series, forecast):
    last_train = series.index[-1].to_pydatetime()
    assert all(ts > last_train for ts in forecast.timestamps)


def test_timestamps_daily_spacing(forecast):
    # Consecutive forecast timestamps should be one day apart
    from datetime import timedelta
    diffs = [forecast.timestamps[i+1] - forecast.timestamps[i] for i in range(HORIZON-1)]
    assert all(d == timedelta(days=1) for d in diffs)


# ── timing and metadata ───────────────────────────────────────────────────────

def test_fit_time_is_positive(forecast):
    assert forecast.fit_time_seconds > 0


def test_metadata_has_aic(forecast):
    assert "aic" in forecast.metadata
    assert isinstance(forecast.metadata["aic"], float)


def test_metadata_records_seasonal_periods(forecast):
    assert forecast.metadata["seasonal_periods"] == 7


def test_metadata_no_fallback(forecast):
    assert forecast.metadata["seasonal_fallback"] is False


# ── seasonal fallback ─────────────────────────────────────────────────────────

def test_short_series_falls_back_to_nonseasonal():
    """Series of 10 points is shorter than 2×7=14 required for weekly ETS."""
    n   = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    s   = pd.Series(np.ones(n) * 5.0 + RNG.normal(0, 0.1, n), index=idx)
    fc  = ETSForecaster(seasonal_periods=7, seasonal="add")
    fc.fit(s)
    assert fc._fell_back is True
    assert fc._effective_seasonal is None


def test_short_series_still_produces_forecast():
    n   = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    s   = pd.Series(np.ones(n) * 5.0 + RNG.normal(0, 0.1, n), index=idx)
    fc  = ETSForecaster(seasonal_periods=7, seasonal="add")
    fc.fit(s)
    r   = fc.predict(7)
    assert len(r.point_forecast) == 7


# ── frequency inference ───────────────────────────────────────────────────────

def test_infers_weekly_period_from_daily_index(series):
    fc = ETSForecaster(seasonal_periods=None, seasonal="add")   # let it infer
    fc.fit(series)
    # Daily frequency → should infer period=7
    assert fc._effective_periods == 7


# ── guard rails ───────────────────────────────────────────────────────────────

def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        ETSForecaster().predict(7)


def test_integer_index_raises():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        ETSForecaster().fit(pd.Series([1.0, 2.0, 3.0]))


def test_nan_in_series_raises():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    s   = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(ValueError, match="NaN"):
        ETSForecaster().fit(s)


# ── model card ────────────────────────────────────────────────────────────────

def test_card_name():
    assert ETSForecaster().card.name == "ets"


def test_card_does_not_handle_missing():
    assert ETSForecaster().card.handles_missing is False


def test_card_supports_native_uncertainty():
    assert ETSForecaster().card.supports_uncertainty == "native"
