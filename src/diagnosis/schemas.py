from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class DetectedFrequency(str, Enum):
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    irregular = "irregular"


class DateRange(BaseModel):
    start: str
    end: str


class ColumnStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    skew: float
    kurtosis: float


class ColumnDiagnostic(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    is_numeric: bool
    is_categorical: bool
    stats: Optional[ColumnStats] = None


class StationarityResult(BaseModel):
    is_stationary: bool
    adf_pvalue: Optional[float]    # None when the series is too short to test
    kpss_pvalue: Optional[float]
    recommended_differencing: int  # 0, 1, or 2
    note: Optional[str] = None     # populated for edge cases (constant, too short)


class SeasonalityResult(BaseModel):
    has_seasonality: bool
    detected_periods: list[int]      # integer periods in samples, e.g. [7, 14]
    seasonality_strength: float      # 0–1: seasonal power / total detrended power
    dominant_period: Optional[int]   # period with highest FFT power among detected


class AnomalyDetail(BaseModel):
    index: int          # positional index in the original series
    value: float
    score: float        # z-score for zscore method, anomaly score for isolation forest


class AnomalyResult(BaseModel):
    anomaly_count: int
    anomaly_indices: list[int]
    method_used: str            # "zscore" or "isolation_forest"
    details: list[AnomalyDetail]


class DiagnosisReport(BaseModel):
    datetime_column: Optional[str]   # None when datetime lives in the index
    datetime_in_index: bool
    numeric_columns: list[str]
    categorical_columns: list[str]
    row_count: int
    date_range: Optional[DateRange]
    detected_frequency: DetectedFrequency
    duplicate_timestamps: int
    columns: list[ColumnDiagnostic]


class FullDiagnosisReport(BaseModel):
    target_column: str
    profile: DiagnosisReport
    stationarity: StationarityResult
    seasonality: SeasonalityResult
    anomalies: AnomalyResult

    def to_summary(self) -> str:
        """Render a human-readable 6-line digest suitable for passing to an LLM agent."""
        p    = self.profile
        stat = self.stationarity
        seas = self.seasonality
        anom = self.anomalies

        # ── line 1: dataset overview ─────────────────────────────────────────
        date_part = ""
        if p.date_range:
            start = _fmt_month(p.date_range.start)
            end   = _fmt_month(p.date_range.end)
            date_part = f", {start} to {end}"
        l1 = f"Dataset: {p.row_count} rows, {p.detected_frequency.value} frequency{date_part}."

        # ── line 2: target column stats ──────────────────────────────────────
        target_diag = next((c for c in p.columns if c.name == self.target_column), None)
        if target_diag and target_diag.stats:
            s  = target_diag.stats
            l2 = f"Target column: '{self.target_column}' (mean: {s.mean:.1f}, std: {s.std:.1f})."
        else:
            l2 = f"Target column: '{self.target_column}'."

        # ── line 3: missing values ───────────────────────────────────────────
        if target_diag and target_diag.missing_count > 0:
            l3 = f"{target_diag.missing_count} missing values detected ({target_diag.missing_pct:.1f}%)."
        else:
            l3 = "No missing values in target column."

        # ── line 4: stationarity ─────────────────────────────────────────────
        if stat.adf_pvalue is None:
            l4 = f"Stationarity test skipped ({stat.note})."
        elif stat.is_stationary:
            l4 = f"Series is stationary (ADF p={stat.adf_pvalue:.2f}, KPSS p={stat.kpss_pvalue:.2f})."
        else:
            d_word = {1: "First", 2: "Second"}.get(stat.recommended_differencing, "Higher-order")
            l4 = f"Series is non-stationary (ADF p={stat.adf_pvalue:.2f}). {d_word} differencing recommended."

        # ── line 5: seasonality ──────────────────────────────────────────────
        if seas.has_seasonality:
            strength_label = (
                "Strong"   if seas.seasonality_strength >= 0.6 else
                "Moderate" if seas.seasonality_strength >= 0.3 else
                "Weak"
            )
            l5 = f"{strength_label} seasonality detected (period={seas.dominant_period}, strength={seas.seasonality_strength:.2f})."
        else:
            l5 = "No clear seasonality detected."

        # ── line 6: anomalies ────────────────────────────────────────────────
        if anom.anomaly_count > 0:
            l6 = f"{anom.anomaly_count} anomalies flagged ({anom.method_used} method)."
        else:
            l6 = "No anomalies detected."

        return "\n".join([l1, l2, l3, l4, l5, l6])


def _fmt_month(iso: str) -> str:
    """'2024-01-01T00:00:00' → 'Jan 2024'"""
    return datetime.fromisoformat(iso).strftime("%b %Y")
