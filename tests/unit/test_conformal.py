"""
Tests for ConformalWrapper and the _conformal_quantile helper.

Coverage tests: the key statistical claim is that conformal prediction gives
≥ (1-α) coverage with finite samples. We test this on 50 held-out points
predicted by a ConformalWrapper(ProphetForecaster). With 90 calibration points,
the quantile formula guarantees coverage ≥ nominal; we allow a ±10 % band
around the target to account for empirical variance over 50 test points.

MAPIEWrapper tests are skipped when MAPIE is not installed.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting.conformal import (
    ConformalWrapper,
    MAPIEWrapper,
    _conformal_quantile,
    _MAPIE_AVAILABLE,
)
from forecasting.prophet_model import ProphetForecaster
from forecasting.schemas import ForecastResult

RNG     = np.random.default_rng(99)
HORIZON = 30


# ── synthetic data ────────────────────────────────────────────────────────────

def _make_series(n: int) -> pd.Series:
    """Daily series: linear trend + strong weekly cycle + low noise."""
    t     = np.arange(n)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    y = (
        80.0
        + 0.10 * t                              # trend keeps series non-stationary
        + 10.0 * np.sin(2 * np.pi * t / 7)     # weekly seasonality, amplitude >> noise
        + RNG.normal(0, 1.5, n)
    )
    return pd.Series(y, index=dates)


# 400 points: 350 for wrapper.fit(), 50 held out for coverage evaluation
_FULL   = _make_series(400)
_FIT    = _FULL.iloc[:350]      # wrapper will train on 70% of this (245 pts)
_ACTUAL = _FULL.iloc[350:].values


@pytest.fixture(scope="module")
def wrapper() -> ConformalWrapper:
    """ConformalWrapper fitted once and shared across the whole test module."""
    fc = ProphetForecaster(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    w = ConformalWrapper(fc)
    w.fit(_FIT)
    return w


@pytest.fixture(scope="module")
def forecast(wrapper) -> ForecastResult:
    return wrapper.predict(HORIZON)


# ── _conformal_quantile unit tests ────────────────────────────────────────────

def test_quantile_returns_correct_order_stat():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # already sorted
    # alpha=0.0 → level = ceil(1.0 * 6) = 6 → scores[5-1] = scores[4] = 5.0
    assert _conformal_quantile(scores, 5, alpha=0.0) == 5.0


def test_quantile_80_percent():
    scores = np.sort(RNG.uniform(0, 10, 100))
    q = _conformal_quantile(scores, 100, alpha=0.20)
    # level = ceil(0.80 * 101) = ceil(80.8) = 81 → scores[80]
    assert q == scores[80]


def test_quantile_insufficient_data_returns_inf():
    scores = np.array([1.0, 2.0])
    # alpha=0.05, level = ceil(0.95 * 3) = ceil(2.85) = 3 > n=2
    assert _conformal_quantile(scores, 2, alpha=0.05) == float("inf")


def test_quantile_monotone_in_alpha():
    scores = np.sort(RNG.uniform(0, 10, 200))
    q80 = _conformal_quantile(scores, 200, alpha=0.20)
    q95 = _conformal_quantile(scores, 200, alpha=0.05)
    # tighter coverage (smaller alpha) → larger quantile → wider interval
    assert q95 >= q80


# ── ConformalWrapper structure ────────────────────────────────────────────────

def test_returns_forecast_result(forecast):
    assert isinstance(forecast, ForecastResult)


def test_model_name_contains_base(forecast):
    assert "prophet" in forecast.model_name
    assert "conformal" in forecast.model_name


def test_timestamps_length(forecast):
    assert len(forecast.timestamps) == HORIZON


def test_all_output_lengths(forecast):
    assert len(forecast.point_forecast) == HORIZON
    assert len(forecast.lower_80) == HORIZON
    assert len(forecast.upper_80) == HORIZON
    assert len(forecast.lower_95) == HORIZON
    assert len(forecast.upper_95) == HORIZON


# ── interval ordering ─────────────────────────────────────────────────────────

def test_lower_80_below_upper_80(forecast):
    assert all(lo < hi for lo, hi in zip(forecast.lower_80, forecast.upper_80))


def test_lower_95_below_upper_95(forecast):
    assert all(lo < hi for lo, hi in zip(forecast.lower_95, forecast.upper_95))


def test_point_inside_80_interval(forecast):
    assert all(
        lo <= pt <= hi
        for lo, pt, hi in zip(forecast.lower_80, forecast.point_forecast, forecast.upper_80)
    )


def test_point_inside_95_interval(forecast):
    assert all(
        lo <= pt <= hi
        for lo, pt, hi in zip(forecast.lower_95, forecast.point_forecast, forecast.upper_95)
    )


def test_95_interval_at_least_as_wide_as_80(forecast):
    widths_80 = [hi - lo for lo, hi in zip(forecast.lower_80, forecast.upper_80)]
    widths_95 = [hi - lo for lo, hi in zip(forecast.lower_95, forecast.upper_95)]
    assert all(w95 >= w80 for w95, w80 in zip(widths_95, widths_80))


def test_intervals_are_symmetric_around_point(forecast):
    # conformal wrapping adds/subtracts the same q from every step
    hw80 = [
        (hi - pt, pt - lo)
        for lo, pt, hi in zip(forecast.lower_80, forecast.point_forecast, forecast.upper_80)
    ]
    assert all(abs(upper - lower) < 1e-9 for upper, lower in hw80)


# ── metadata ─────────────────────────────────────────────────────────────────

def test_metadata_has_n_calibration(forecast):
    assert "n_calibration" in forecast.metadata
    # 30% of 350 = 105 calibration points
    assert forecast.metadata["n_calibration"] == 105


def test_metadata_has_quantiles(forecast):
    assert "q80" in forecast.metadata
    assert "q95" in forecast.metadata
    assert forecast.metadata["q95"] >= forecast.metadata["q80"]


# ── empirical coverage ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def coverage_forecast(wrapper) -> ForecastResult:
    """50-step forecast for comparing against held-out actuals."""
    return wrapper.predict(50)


def test_empirical_coverage_80(coverage_forecast):
    lo = np.array(coverage_forecast.lower_80)
    hi = np.array(coverage_forecast.upper_80)
    coverage = np.mean((lo <= _ACTUAL) & (_ACTUAL <= hi))
    # conformal guarantee: coverage ≥ 0.80; allow ±10 % for empirical variance
    assert 0.70 <= coverage <= 0.90, f"80% coverage was {coverage:.2%}"


def test_empirical_coverage_95(coverage_forecast):
    lo = np.array(coverage_forecast.lower_95)
    hi = np.array(coverage_forecast.upper_95)
    coverage = np.mean((lo <= _ACTUAL) & (_ACTUAL <= hi))
    # allow ±10 % around 95 %
    assert 0.85 <= coverage <= 1.00, f"95% coverage was {coverage:.2%}"


# ── guard rails ───────────────────────────────────────────────────────────────

def test_predict_before_fit_raises():
    w = ConformalWrapper(ProphetForecaster())
    with pytest.raises(RuntimeError, match="not been fitted"):
        w.predict(7)


def test_card_inherits_base_best_for():
    w = ConformalWrapper(ProphetForecaster())
    assert "seasonality" in w.card.best_for.lower()


def test_card_supports_conformal():
    w = ConformalWrapper(ProphetForecaster())
    assert w.card.supports_uncertainty == "conformal"


# ── MAPIEWrapper (skipped when MAPIE absent) ──────────────────────────────────

@pytest.mark.skipif(not _MAPIE_AVAILABLE, reason="MAPIE not installed")
def test_mapie_wrapper_produces_forecast_result():
    fc = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    w  = MAPIEWrapper(fc)
    w.fit(_FIT)
    r  = w.predict(14)
    assert isinstance(r, ForecastResult)
    assert len(r.point_forecast) == 14


@pytest.mark.skipif(not _MAPIE_AVAILABLE, reason="MAPIE not installed")
def test_mapie_wrapper_interval_ordering():
    fc = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    w  = MAPIEWrapper(fc)
    w.fit(_FIT)
    r  = w.predict(14)
    assert all(lo <= pt <= hi for lo, pt, hi in zip(r.lower_80, r.point_forecast, r.upper_80))
    assert all(lo <= pt <= hi for lo, pt, hi in zip(r.lower_95, r.point_forecast, r.upper_95))


def test_mapie_wrapper_raises_without_mapie(monkeypatch):
    import forecasting.conformal as cm
    monkeypatch.setattr(cm, "_MAPIE_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install mapie"):
        MAPIEWrapper(ProphetForecaster())
