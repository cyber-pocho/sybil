from __future__ import annotations

import pandas as pd

from diagnosis.schemas import (
    ColumnDiagnostic,
    ColumnStats,
    DateRange,
    DetectedFrequency,
    DiagnosisReport,
)

# Median-delta thresholds in seconds for each named frequency
_FREQ_BANDS: list[tuple[float, float, DetectedFrequency]] = [
    (1_800,    5_400,   DetectedFrequency.hourly),
    (72_000,   108_000, DetectedFrequency.daily),
    (518_400,  691_200, DetectedFrequency.weekly),
    (2_246_400, 2_851_200, DetectedFrequency.monthly),
]

# Max tolerated MAD/median ratio of inter-row gaps before a series is called
# irregular. 0 for perfectly regular data; stays 0 for a daily series with
# weekend gaps, since MAD ignores a minority of outlying deltas.
_MAX_DELTA_DISPERSION = 0.25


class DataProfiler:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_datetime(self) -> tuple[str | None, bool]:
        """Return (column_name, is_in_index).

        Checks the index first, then probes object/string columns with
        pd.to_datetime; a column qualifies when ≥80 % of rows parse cleanly.
        """
        if isinstance(self.df.index, pd.DatetimeIndex):
            return (self.df.index.name, True)

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            try:
                parsed = pd.to_datetime(self.df[col], errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    return (col, False)
            except Exception:
                continue

        return (None, False)

    def _as_datetime_series(
        self, dt_col: str | None, in_index: bool
    ) -> pd.Series | None:
        if in_index:
            return pd.Series(self.df.index, name=dt_col)
        if dt_col is not None:
            return pd.to_datetime(self.df[dt_col], errors="coerce")
        return None

    @staticmethod
    def _detect_frequency(dt: pd.Series) -> DetectedFrequency:
        sorted_dt = dt.dropna().sort_values()
        if len(sorted_dt) < 2:
            return DetectedFrequency.irregular

        deltas = sorted_dt.diff().dropna().dt.total_seconds()
        median_seconds = float(deltas.median())

        if median_seconds <= 0:
            return DetectedFrequency.irregular   # all timestamps duplicated

        # The median alone says nothing about how *regularly* spaced the rows are:
        # deltas of 2 d, 7 d and 36 d have a median of exactly 7 d and would be
        # reported as clean weekly data. Gate on the median absolute deviation to
        # catch that. MAD is used rather than the standard deviation because it
        # ignores a minority of odd gaps — a daily series that skips weekends has
        # MAD 0 and should still read as daily.
        mad_seconds = float((deltas - median_seconds).abs().median())
        if mad_seconds / median_seconds > _MAX_DELTA_DISPERSION:
            return DetectedFrequency.irregular

        for lo, hi, freq in _FREQ_BANDS:
            if lo <= median_seconds <= hi:
                return freq
        return DetectedFrequency.irregular

    @staticmethod
    def _column_stats(series: pd.Series) -> ColumnStats:
        valid = series.dropna()
        return ColumnStats(
            mean=float(valid.mean()),
            std=float(valid.std(ddof=1)),
            min=float(valid.min()),
            max=float(valid.max()),
            skew=float(valid.skew()),
            kurtosis=float(valid.kurtosis()),  # excess kurtosis, same as scipy default
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> DiagnosisReport:
        df = self.df
        n = len(df)

        dt_col, in_index = self._find_datetime()
        dt_series = self._as_datetime_series(dt_col, in_index)

        # Columns available for analysis (exclude the datetime column if it
        # sits in the DataFrame proper rather than the index).
        analysis_cols = [c for c in df.columns if c != dt_col or in_index]

        numeric_cols = [
            c for c in analysis_cols if pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_cols = [
            c for c in analysis_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]

        date_range: DateRange | None = None
        detected_freq = DetectedFrequency.irregular
        dup_timestamps = 0

        if dt_series is not None:
            date_range = DateRange(
                start=dt_series.min().isoformat(),
                end=dt_series.max().isoformat(),
            )
            detected_freq = self._detect_frequency(dt_series)
            dup_timestamps = int(dt_series.duplicated().sum())

        col_diagnostics: list[ColumnDiagnostic] = []
        for col in analysis_cols:
            s = df[col]
            missing = int(s.isna().sum())
            is_num = pd.api.types.is_numeric_dtype(s)
            col_diagnostics.append(
                ColumnDiagnostic(
                    name=col,
                    dtype=str(s.dtype),
                    missing_count=missing,
                    missing_pct=round(missing / n * 100, 2) if n else 0.0,
                    is_numeric=is_num,
                    is_categorical=not is_num,
                    stats=self._column_stats(s) if is_num else None,
                )
            )

        return DiagnosisReport(
            datetime_column=dt_col if not in_index else None,
            datetime_in_index=in_index,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            row_count=n,
            date_range=date_range,
            detected_frequency=detected_freq,
            duplicate_timestamps=dup_timestamps,
            columns=col_diagnostics,
        )


def profile(df: pd.DataFrame) -> DiagnosisReport:
    return DataProfiler(df).run()
