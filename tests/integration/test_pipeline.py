"""
Integration test for run_full_diagnosis().

Uses a single synthetic DataFrame that has every property the pipeline is
designed to detect: linear trend (non-stationarity), weekly seasonality,
injected missing values, and three large spike anomalies.

These tests are slower than unit tests (~2-5 s for stationarity + IF) so they
live in tests/integration/ and are excluded from the CI unit-test run.
Run them locally with: pytest tests/integration/ -v
"""

import numpy as np
import pandas as pd
import pytest

from diagnosis.pipeline import run_full_diagnosis
from diagnosis.schemas import FullDiagnosisReport

RNG = np.random.default_rng(7)

SPIKE_POSITIONS = [50, 200, 500]   # known anomaly locations
MISSING_POSITIONS = [10, 100, 300, 450, 600]


@pytest.fixture(scope="module")   # build the DataFrame and run the pipeline once for all tests
def report() -> FullDiagnosisReport:
    n = 730   # two years of daily observations
    t = np.arange(n)

    trend      = 0.15 * t                              # slow upward drift → non-stationary
    seasonality = 8.0 * np.sin(2 * np.pi * t / 7)    # weekly cycle, amplitude well above noise
    noise      = RNG.normal(0, 2.0, n)
    signal     = 100.0 + trend + seasonality + noise

    # Inject obvious spikes so anomaly detection has clear targets
    signal[SPIKE_POSITIONS] = 500.0

    # Introduce missing values
    signal = signal.astype(float)
    signal[MISSING_POSITIONS] = np.nan

    df = pd.DataFrame({
        "date":  pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "sales": signal,
    })

    return run_full_diagnosis(df)


# ── return type ───────────────────────────────────────────────────────────────

def test_returns_full_diagnosis_report(report):
    assert isinstance(report, FullDiagnosisReport)


def test_target_column_is_sales(report):
    assert report.target_column == "sales"


# ── profile sub-report ────────────────────────────────────────────────────────

def test_profile_row_count(report):
    assert report.profile.row_count == 730


def test_profile_detects_daily_frequency(report):
    assert report.profile.detected_frequency.value == "daily"


def test_profile_detects_missing_values(report):
    sales_diag = next(c for c in report.profile.columns if c.name == "sales")
    assert sales_diag.missing_count == len(MISSING_POSITIONS)


# ── stationarity sub-report ───────────────────────────────────────────────────

def test_trend_series_is_non_stationary(report):
    # Linear trend → ADF should fail to reject unit root
    assert report.stationarity.is_stationary is False


def test_recommends_differencing(report):
    assert report.stationarity.recommended_differencing >= 1


def test_stationarity_pvalues_populated(report):
    assert report.stationarity.adf_pvalue is not None
    assert report.stationarity.kpss_pvalue is not None


# ── seasonality sub-report ────────────────────────────────────────────────────

def test_weekly_seasonality_detected(report):
    assert report.seasonality.has_seasonality is True


def test_dominant_period_is_weekly(report):
    assert report.seasonality.dominant_period == 7


def test_seasonality_strength_is_meaningful(report):
    assert report.seasonality.seasonality_strength > 0.1


# ── anomaly sub-report ────────────────────────────────────────────────────────

def test_anomalies_detected(report):
    assert report.anomalies.anomaly_count > 0


def test_spike_positions_flagged(report):
    # All three injected spikes must appear in the flagged indices
    found = set(report.anomalies.anomaly_indices)
    assert set(SPIKE_POSITIONS).issubset(found)


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_is_nonempty(report):
    summary = report.to_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_summary_has_six_lines(report):
    lines = report.to_summary().strip().splitlines()
    assert len(lines) == 6


def test_summary_mentions_row_count(report):
    assert "730" in report.to_summary()


def test_summary_mentions_target_column(report):
    assert "sales" in report.to_summary()


def test_summary_mentions_non_stationary(report):
    assert "non-stationary" in report.to_summary()


def test_summary_mentions_seasonality(report):
    summary = report.to_summary()
    assert "seasonality" in summary.lower()


def test_summary_mentions_anomalies(report):
    summary = report.to_summary()
    assert "anomal" in summary.lower()


# ── explicit target column ────────────────────────────────────────────────────

def test_explicit_target_column_accepted():
    df = pd.DataFrame({
        "date":   pd.date_range("2024-01-01", periods=60, freq="D").astype(str),
        "revenue": np.ones(60) * 10,
        "cost":    np.ones(60) * 5,
    })
    r = run_full_diagnosis(df, target_column="cost")
    assert r.target_column == "cost"


def test_invalid_target_column_raises():
    df = pd.DataFrame({"date": ["2024-01-01"], "sales": [1.0]})
    with pytest.raises(ValueError, match="not found"):
        run_full_diagnosis(df, target_column="nonexistent")


def test_no_numeric_columns_raises():
    df = pd.DataFrame({"date": ["2024-01-01"], "label": ["A"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        run_full_diagnosis(df)
