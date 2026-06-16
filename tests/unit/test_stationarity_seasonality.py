import numpy as np
import pandas as pd
import pytest

from diagnosis.seasonality import detect_seasonality
from diagnosis.schemas import SeasonalityResult, StationarityResult
from diagnosis.stationarity import test_stationarity

RNG = np.random.default_rng(42)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def stationary_series() -> pd.Series:
    """White noise: zero mean, constant variance — textbook stationary."""
    return pd.Series(RNG.normal(0, 1, 300))


@pytest.fixture()
def trend_series() -> pd.Series:
    """Random walk (cumulative sum of noise): I(1), has a unit root."""
    return pd.Series(RNG.normal(0, 1, 300).cumsum())


@pytest.fixture()
def weekly_seasonal_series() -> pd.Series:
    """Two years of daily data with a strong 7-sample (weekly) sine wave."""
    t = np.arange(365 * 2)
    signal = 6.0 * np.sin(2 * np.pi * t / 7)   # amplitude well above noise
    noise  = RNG.normal(0, 0.8, len(t))
    return pd.Series(signal + noise)


@pytest.fixture()
def flat_noise_series() -> pd.Series:
    """Pure white noise: no periodic structure, seasonality detector should find nothing."""
    return pd.Series(RNG.normal(0, 1, 200))


# ── stationarity ─────────────────────────────────────────────────────────────

def test_white_noise_is_stationary(stationary_series):
    r = test_stationarity(stationary_series)
    assert r.is_stationary is True
    assert r.recommended_differencing == 0
    assert r.adf_pvalue < 0.05
    assert r.kpss_pvalue > 0.05


def test_random_walk_is_not_stationary(trend_series):
    r = test_stationarity(trend_series)
    assert r.is_stationary is False


def test_random_walk_recommends_one_difference(trend_series):
    r = test_stationarity(trend_series)
    assert r.recommended_differencing >= 1


def test_too_short_series_returns_safely():
    r = test_stationarity(pd.Series([1.0, 2.0, 3.0]))
    assert r.is_stationary is False
    assert r.adf_pvalue is None
    assert r.kpss_pvalue is None
    assert r.note is not None


def test_constant_series_is_stationary():
    r = test_stationarity(pd.Series([5.0] * 50))
    assert r.is_stationary is True
    assert r.recommended_differencing == 0
    assert "constant" in (r.note or "")


def test_series_with_nans_handled(stationary_series):
    with_nans = stationary_series.copy()
    with_nans.iloc[::10] = np.nan   # 10 % missing
    r = test_stationarity(with_nans)
    assert isinstance(r, StationarityResult)


def test_returns_stationarity_result(stationary_series):
    assert isinstance(test_stationarity(stationary_series), StationarityResult)


# ── seasonality ──────────────────────────────────────────────────────────────

def test_weekly_seasonality_detected(weekly_seasonal_series):
    r = detect_seasonality(weekly_seasonal_series, freq="D")
    assert r.has_seasonality is True
    assert 7 in r.detected_periods


def test_weekly_dominant_period(weekly_seasonal_series):
    r = detect_seasonality(weekly_seasonal_series, freq="D")
    assert r.dominant_period == 7


def test_weekly_strength_is_high(weekly_seasonal_series):
    r = detect_seasonality(weekly_seasonal_series, freq="D")
    assert r.seasonality_strength > 0.5


def test_white_noise_has_no_seasonality(flat_noise_series):
    r = detect_seasonality(flat_noise_series, freq="D")
    assert r.has_seasonality is False
    assert r.detected_periods == []
    assert r.dominant_period is None


def test_white_noise_strength_is_low(flat_noise_series):
    r = detect_seasonality(flat_noise_series, freq="D")
    assert r.seasonality_strength < 0.3


def test_too_short_series_no_seasonality():
    r = detect_seasonality(pd.Series([1.0, 2.0, 3.0]), freq="D")
    assert r.has_seasonality is False
    assert r.seasonality_strength == 0.0


def test_series_with_nans_handled(weekly_seasonal_series):
    with_nans = weekly_seasonal_series.copy()
    with_nans.iloc[::20] = np.nan
    r = detect_seasonality(with_nans, freq="D")
    assert isinstance(r, SeasonalityResult)


def test_returns_seasonality_result(weekly_seasonal_series):
    assert isinstance(detect_seasonality(weekly_seasonal_series, freq="D"), SeasonalityResult)


def test_unknown_freq_falls_back_gracefully(weekly_seasonal_series):
    # An unrecognised freq string should not raise; it defaults to period=7
    r = detect_seasonality(weekly_seasonal_series, freq="UNKNOWN")
    assert isinstance(r, SeasonalityResult)
