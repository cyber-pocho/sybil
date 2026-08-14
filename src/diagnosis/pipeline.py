"""
Full diagnostic pipeline: profile → anomalies → stationarity → seasonality.

run_full_diagnosis() is the single entry point. It picks the first numeric
column as the forecast target, then runs all four diagnosis modules in sequence,
passing results downstream so each step can build on what the previous one found:
the profiler's detected frequency is fed into seasonality, and the anomaly
indices are used to de-spike the series before seasonality measures it.

Anomaly detection therefore comes second, not last. Running it last — after the
steps whose measurements the anomalies distort — meant the pipeline located the
outliers one step too late to do anything about them.
"""

import numpy as np
import pandas as pd

from diagnosis.anomalies import detect_anomalies
from diagnosis.profiler import profile
from diagnosis.schemas import DetectedFrequency, FullDiagnosisReport
from diagnosis.seasonality import detect_seasonality
from diagnosis.stationarity import check_stationarity

# Map the profiler's DetectedFrequency enum to the pandas freq string that
# detect_seasonality expects. "irregular" falls back to "D" (daily) so the
# seasonality module still runs and can return has_seasonality=False cleanly.
_FREQ_TO_PANDAS: dict[DetectedFrequency, str] = {
    DetectedFrequency.hourly:    "h",
    DetectedFrequency.daily:     "D",
    DetectedFrequency.weekly:    "W",
    DetectedFrequency.monthly:   "MS",
    DetectedFrequency.irregular: "D",
}


def run_full_diagnosis(df: pd.DataFrame, target_column: str | None = None) -> FullDiagnosisReport:
    """
    Args:
        df:            Raw DataFrame. May contain NaNs and mixed column types.
        target_column: Column to analyse. Defaults to the first numeric column
                       found by the profiler.

    Returns:
        FullDiagnosisReport with all four sub-reports populated and a
        .to_summary() method for LLM context generation.
    """
    # ── step 1: profile ───────────────────────────────────────────────────────
    profile_report = profile(df)

    if not profile_report.numeric_columns:
        raise ValueError("DataFrame has no numeric columns — nothing to diagnose.")

    # Pick target: caller-supplied name takes precedence, else first numeric column
    target = target_column if target_column else profile_report.numeric_columns[0]
    if target not in df.columns:
        raise ValueError(f"target_column '{target}' not found in DataFrame.")

    series = df[target].reset_index(drop=True)   # align index to 0-based ints for anomaly indices

    # ── step 2: anomalies ─────────────────────────────────────────────────────
    # Runs before the characterisation steps, not after, so that what it finds
    # can be fed downstream — see the cleaned series built below.
    anomaly_result = detect_anomalies(series, method="auto")

    # ── step 3: stationarity ──────────────────────────────────────────────────
    # Deliberately on the raw series: a unit root is a property of the trend
    # rather than of a few spikes, and the differencing recommendation should
    # describe the data the caller actually has.
    stationarity_result = check_stationarity(series)

    # ── step 4: seasonality, on the de-spiked series ──────────────────────────
    # Seasonality is the one step outliers genuinely break: a handful of spikes
    # dominate the FFT power budget and drag the ACF at the true lag below the
    # Bartlett band, so a clean weekly cycle reports as no seasonality at all.
    # Interpolating over the flagged points preserves both the length and the
    # phase of the series — dropping them would shift every later observation
    # and smear the very periodicity we are trying to measure.
    cleaned = series.astype(float).copy()
    cleaned.iloc[anomaly_result.anomaly_indices] = np.nan
    cleaned = cleaned.interpolate(limit_direction="both")

    pandas_freq = _FREQ_TO_PANDAS[profile_report.detected_frequency]
    seasonality_result = detect_seasonality(cleaned, freq=pandas_freq)

    return FullDiagnosisReport(
        target_column=target,
        profile=profile_report,
        stationarity=stationarity_result,
        seasonality=seasonality_result,
        anomalies=anomaly_result,
    )
