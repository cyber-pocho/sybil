import numpy as np
import pandas as pd
import pytest

from diagnosis.anomalies import detect_anomalies, isolation_forest_anomalies, zscore_anomalies
from diagnosis.schemas import AnomalyResult

RNG = np.random.default_rng(0)

# Spike positions injected into fixtures — known ahead of time so tests can assert on them
SPIKE_POSITIONS = [20, 80, 150]
SPIKE_VALUE     = 50.0   # far enough from μ≈0, σ≈1 to be unambiguous


@pytest.fixture()
def clean_series() -> pd.Series:
    """Gaussian white noise: no anomalies at 3σ threshold."""
    return pd.Series(RNG.normal(0, 1, 200))


@pytest.fixture()
def spiked_series() -> pd.Series:
    """Same noise with three large positive spikes injected at known positions."""
    s = RNG.normal(0, 1, 200).copy()
    s[SPIKE_POSITIONS] = SPIKE_VALUE
    return pd.Series(s)


# ── z-score ───────────────────────────────────────────────────────────────────

def test_zscore_clean_series_no_anomalies(clean_series):
    assert zscore_anomalies(clean_series) == []


def test_zscore_detects_all_spikes(spiked_series):
    found = {d.index for d in zscore_anomalies(spiked_series)}
    assert set(SPIKE_POSITIONS).issubset(found)


def test_zscore_score_magnitude(spiked_series):
    details = zscore_anomalies(spiked_series)
    spike_details = [d for d in details if d.index in SPIKE_POSITIONS]
    # Spikes are ~50σ above the mean; all z-scores should be large
    assert all(d.score > 3.0 for d in spike_details)


def test_zscore_custom_threshold(spiked_series):
    # A threshold higher than any spike z-score should yield nothing
    assert zscore_anomalies(spiked_series, threshold=1000.0) == []


def test_zscore_constant_series_returns_empty():
    assert zscore_anomalies(pd.Series([7.0] * 100)) == []


def test_zscore_nans_ignored():
    s = pd.Series([1.0, np.nan, 2.0, 100.0])
    details = zscore_anomalies(s)
    assert any(d.value == 100.0 for d in details)


# ── isolation forest ──────────────────────────────────────────────────────────

def test_iforest_detects_spikes(spiked_series):
    found = {d.index for d in isolation_forest_anomalies(spiked_series)}
    assert set(SPIKE_POSITIONS).issubset(found)


def test_iforest_clean_series_anomaly_count(clean_series):
    # contamination=0.05 → at most 5 % of 200 = 10 anomalies labelled
    details = isolation_forest_anomalies(clean_series, contamination=0.05)
    assert len(details) <= int(0.05 * len(clean_series)) + 1


def test_iforest_score_is_negative_for_anomalies(spiked_series):
    # IsolationForest decision_function: anomalies score below 0
    spike_details = [
        d for d in isolation_forest_anomalies(spiked_series)
        if d.index in SPIKE_POSITIONS
    ]
    assert all(d.score < 0 for d in spike_details)


# ── detect_anomalies (auto dispatch) ─────────────────────────────────────────

def test_auto_selects_zscore_for_gaussian(clean_series):
    r = detect_anomalies(clean_series, method="auto")
    assert r.method_used == "zscore"


def test_auto_selects_iforest_for_skewed():
    # Log-normal is strongly non-Gaussian → Shapiro-Wilk rejects normality
    skewed = pd.Series(RNG.lognormal(0, 1, 300))
    r = detect_anomalies(skewed, method="auto")
    assert r.method_used == "isolation_forest"


def test_explicit_zscore_method(spiked_series):
    r = detect_anomalies(spiked_series, method="zscore")
    assert r.method_used == "zscore"
    assert set(SPIKE_POSITIONS).issubset(set(r.anomaly_indices))


def test_explicit_iforest_method(spiked_series):
    r = detect_anomalies(spiked_series, method="isolation_forest")
    assert r.method_used == "isolation_forest"


def test_anomaly_count_matches_details(spiked_series):
    r = detect_anomalies(spiked_series)
    assert r.anomaly_count == len(r.details)
    assert r.anomaly_count == len(r.anomaly_indices)


def test_returns_anomaly_result(spiked_series):
    assert isinstance(detect_anomalies(spiked_series), AnomalyResult)


def test_clean_series_low_anomaly_count(clean_series):
    r = detect_anomalies(clean_series, method="zscore")
    # Gaussian at 3σ: expect <1 % false positives in 200 samples
    assert r.anomaly_count <= 2
