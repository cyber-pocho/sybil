import numpy as np
import pandas as pd
import pytest

from diagnosis.profiler import DataProfiler, profile
from diagnosis.schemas import DetectedFrequency, DiagnosisReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daily_df() -> pd.DataFrame:
    """30-day daily DataFrame with two numeric targets, one category, and
    intentional missing values."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.astype(str),          # stored as strings → triggers heuristic
            "revenue": rng.normal(1_000, 150, 30),
            "units": rng.integers(50, 200, 30).astype(float),
            "region": rng.choice(["North", "South", "East"], 30),
        }
    )
    # 3 missing in revenue (~10 %), 1 missing in units
    df.loc[[2, 7, 15], "revenue"] = np.nan
    df.loc[[5], "units"] = np.nan
    return df


@pytest.fixture()
def indexed_df() -> pd.DataFrame:
    """DataFrame whose index is already a DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=10, freq="W")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"value": rng.normal(0, 1, 10)}, index=idx)


# ---------------------------------------------------------------------------
# Datetime detection
# ---------------------------------------------------------------------------

def test_detects_string_datetime_column(daily_df):
    report = profile(daily_df)
    assert report.datetime_column == "date"
    assert report.datetime_in_index is False


def test_detects_datetimeindex(indexed_df):
    report = profile(indexed_df)
    assert report.datetime_in_index is True
    assert report.datetime_column is None   # lives in the index, not a column


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------

def test_daily_frequency(daily_df):
    assert profile(daily_df).detected_frequency == DetectedFrequency.daily


def test_weekly_frequency(indexed_df):
    assert profile(indexed_df).detected_frequency == DetectedFrequency.weekly


def test_hourly_frequency():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=24, freq="h").astype(str),
            "val": np.ones(24),
        }
    )
    assert profile(df).detected_frequency == DetectedFrequency.hourly


def test_monthly_frequency():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=12, freq="MS").astype(str),
            "val": np.ones(12),
        }
    )
    assert profile(df).detected_frequency == DetectedFrequency.monthly


def test_irregular_frequency():
    df = pd.DataFrame(
        {
            "ts": ["2024-01-01", "2024-01-03", "2024-01-10", "2024-02-15"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert profile(df).detected_frequency == DetectedFrequency.irregular


# ---------------------------------------------------------------------------
# Row count & date range
# ---------------------------------------------------------------------------

def test_row_count(daily_df):
    assert profile(daily_df).row_count == 30


def test_date_range(daily_df):
    dr = profile(daily_df).date_range
    assert dr is not None
    assert dr.start.startswith("2024-01-01")
    assert dr.end.startswith("2024-01-30")


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def test_numeric_columns(daily_df):
    report = profile(daily_df)
    assert set(report.numeric_columns) == {"revenue", "units"}


def test_categorical_columns(daily_df):
    report = profile(daily_df)
    assert report.categorical_columns == ["region"]


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def test_missing_count_revenue(daily_df):
    report = profile(daily_df)
    diag = next(c for c in report.columns if c.name == "revenue")
    assert diag.missing_count == 3


def test_missing_pct_revenue(daily_df):
    report = profile(daily_df)
    diag = next(c for c in report.columns if c.name == "revenue")
    assert diag.missing_pct == pytest.approx(10.0)


def test_missing_count_units(daily_df):
    report = profile(daily_df)
    diag = next(c for c in report.columns if c.name == "units")
    assert diag.missing_count == 1


# ---------------------------------------------------------------------------
# Numeric stats
# ---------------------------------------------------------------------------

def test_numeric_stats_present(daily_df):
    report = profile(daily_df)
    diag = next(c for c in report.columns if c.name == "revenue")
    assert diag.stats is not None
    s = diag.stats
    assert s.min < s.mean < s.max
    assert s.std > 0


def test_categorical_has_no_stats(daily_df):
    report = profile(daily_df)
    diag = next(c for c in report.columns if c.name == "region")
    assert diag.stats is None


# ---------------------------------------------------------------------------
# Duplicate timestamps
# ---------------------------------------------------------------------------

def test_no_duplicate_timestamps(daily_df):
    assert profile(daily_df).duplicate_timestamps == 0


def test_duplicate_timestamp_detection():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "val": [1.0, 2.0, 3.0],
        }
    )
    assert profile(df).duplicate_timestamps == 1


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_diagnosis_report(daily_df):
    assert isinstance(profile(daily_df), DiagnosisReport)


def test_dataprofile_run_equivalent(daily_df):
    assert profile(daily_df) == DataProfiler(daily_df).run()
