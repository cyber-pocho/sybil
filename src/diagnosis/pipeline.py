"""
Full diagnostic pipeline: profile → stationarity → seasonality → anomalies.

run_full_diagnosis() is the single entry point. It picks the first numeric
column as the forecast target, then runs all four diagnosis modules in sequence,
passing results downstream so each step can build on what the previous one found
(e.g. detected frequency from the profiler is fed straight into seasonality).
"""

import pandas as pd

from diagnosis.anomalies import detect_anomalies
from diagnosis.profiler import profile
from diagnosis.schemas import DetectedFrequency, FullDiagnosisReport
from diagnosis.seasonality import detect_seasonality
from diagnosis.stationarity import test_stationarity

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

    # ── step 2: stationarity ──────────────────────────────────────────────────
    stationarity_result = test_stationarity(series)

    # ── step 3: seasonality ───────────────────────────────────────────────────
    pandas_freq = _FREQ_TO_PANDAS[profile_report.detected_frequency]
    seasonality_result = detect_seasonality(series, freq=pandas_freq)

    # ── step 4: anomalies ─────────────────────────────────────────────────────
    anomaly_result = detect_anomalies(series, method="auto")

    return FullDiagnosisReport(
        target_column=target,
        profile=profile_report,
        stationarity=stationarity_result,
        seasonality=seasonality_result,
        anomalies=anomaly_result,
    )
