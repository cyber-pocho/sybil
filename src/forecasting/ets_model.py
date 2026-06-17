"""
ETS (Error-Trend-Seasonal) forecaster backed by statsmodels' ETSModel.

ETS is a state-space generalisation of Holt-Winters exponential smoothing.
Each component can be additive, multiplicative, or absent:
    y(t) = (level + trend) * seasonal + error   ← multiplicative example

We use ETSModel rather than the older ExponentialSmoothing class because it
exposes get_prediction(), which gives analytical prediction intervals without
simulation — the same alpha-based API used by SARIMAX and other state-space
models in statsmodels.

Seasonal fallback chain (in order):
  1. Use caller-supplied seasonal_periods if given.
  2. Infer from the DatetimeIndex frequency (e.g. daily → 7 for weekly cycle).
  3. If the series is shorter than 2 × seasonal_periods, drop seasonality so
     the model can still fit on short data rather than raising an error.
"""

import time
import warnings
from typing import Literal

import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from forecasting.base import BaseForecaster
from forecasting.schemas import ForecastResult, ModelCard

# Maps the first one or two characters of a pandas freq string to the number
# of observations in one natural cycle at that resolution.
_FREQ_TO_PERIOD: dict[str, int] = {
    "T":  60,   # minutely data  → 60-sample hour
    "min": 60,
    "h":  24,   # hourly data    → 24-sample day
    "H":  24,
    "D":  7,    # daily data     → 7-sample week
    "W":  52,   # weekly data    → 52-sample year
    "M":  12,   # monthly data   → 12-sample year
    "Q":  4,    # quarterly data → 4-sample year
}

ErrorType    = Literal["add", "mul"]
TrendType    = Literal["add", "mul"] | None
SeasonalType = Literal["add", "mul"] | None


class ETSForecaster(BaseForecaster):

    def __init__(
        self,
        seasonal_periods: int | None = None,
        error:   ErrorType    = "add",
        trend:   TrendType    = "add",
        seasonal: SeasonalType = "add",
        damped_trend: bool    = True,
    ) -> None:
        self._seasonal_periods = seasonal_periods
        self._error            = error
        self._trend            = trend
        self._seasonal         = seasonal
        self._damped_trend     = damped_trend
        self._result           = None   # holds the fitted ETSResults object
        self._fit_time: float  = 0.0
        # stored after fit so predict() can generate future timestamps
        self._series_index: pd.DatetimeIndex | None = None
        # what actually got used (may differ from __init__ args if we fell back)
        self._effective_periods: int | None  = None
        self._effective_seasonal: SeasonalType = seasonal
        self._fell_back: bool = False

    # ── BaseForecaster contract ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ets"

    @property
    def card(self) -> ModelCard:
        return ModelCard(
            name="ets",
            description=(
                "State-space ETS (Error-Trend-Seasonal) model. "
                "Generalises Holt-Winters with additive or multiplicative components. "
                "Fits analytically via maximum likelihood; prediction intervals are exact."
            ),
            best_for="Medium-length series (50–2000 obs) with stable trend and seasonality, no regressors needed.",
            handles_missing=False,    # statsmodels ETSModel requires a complete series
            min_samples_required=24,  # need at least a couple of seasonal cycles to identify components
            supports_uncertainty="native",
        )

    def fit(self, series: pd.Series, **kwargs) -> None:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(
                "ETSForecaster requires a DatetimeIndex. "
                "Pass series.set_index(<date_column>) before calling fit()."
            )
        if series.isna().any():
            raise ValueError(
                "ETSForecaster does not handle NaNs — drop or interpolate missing values first."
            )

        self._series_index = series.index

        sp, effective_seasonal = self._resolve_seasonal(series)
        self._effective_periods  = sp
        self._effective_seasonal = effective_seasonal

        model = ETSModel(
            series,
            error=self._error,
            trend=self._trend,
            seasonal=effective_seasonal,
            seasonal_periods=sp,
            damped_trend=self._damped_trend if self._trend is not None else False,
        )

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress convergence and frequency warnings
            self._result = model.fit(disp=False, **kwargs)
        self._fit_time = time.perf_counter() - t0

        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        """Return point forecasts with 80 % and 95 % prediction intervals.

        get_prediction() uses the analytical state-space posterior; alpha is
        the *exclusion* probability so alpha=0.20 → 80 % interval,
        alpha=0.05 → 95 % interval.
        """
        self._validate_is_fitted()

        n = len(self._series_index)

        # Integer positions: start at the first out-of-sample step
        pred_80 = self._result.get_prediction(start=n, end=n + horizon - 1)
        pred_95 = self._result.get_prediction(start=n, end=n + horizon - 1)

        # summary_frame(alpha) returns: mean, mean_se, mean_ci_lower, mean_ci_upper
        frame_80 = pred_80.summary_frame(alpha=0.20)
        frame_95 = pred_95.summary_frame(alpha=0.05)

        # Generate future timestamps from the index frequency
        freq = self._series_index.freq or pd.tseries.frequencies.to_offset(
            pd.infer_freq(self._series_index)
        )
        future_index = pd.date_range(start=self._series_index[-1], periods=horizon + 1, freq=freq)[1:]

        return ForecastResult(
            timestamps=future_index.tolist(),
            point_forecast=frame_80["mean"].tolist(),
            lower_80=frame_80["mean_ci_lower"].tolist(),
            upper_80=frame_80["mean_ci_upper"].tolist(),
            lower_95=frame_95["mean_ci_lower"].tolist(),
            upper_95=frame_95["mean_ci_upper"].tolist(),
            model_name=self.name,
            fit_time_seconds=round(self._fit_time, 3),
            metadata={
                "error":            self._error,
                "trend":            self._trend,
                "seasonal":         self._effective_seasonal,
                "damped_trend":     self._damped_trend,
                "seasonal_periods": self._effective_periods,
                "seasonal_fallback": self._fell_back,
                "aic":              round(float(self._result.aic), 3),
            },
        )

    # ── internal ─────────────────────────────────────────────────────────────

    def _resolve_seasonal(self, series: pd.Series) -> tuple[int | None, SeasonalType]:
        """Return (seasonal_periods, seasonal_type) accounting for series length.

        Falls back to non-seasonal when the series is too short — ETSModel
        requires at least 2 × seasonal_periods observations to identify the
        seasonal component.
        """
        sp = self._seasonal_periods

        if sp is None and self._seasonal is not None:
            # Try to infer from the DatetimeIndex frequency
            raw_freq = pd.infer_freq(series.index) or ""
            # Match against known prefixes (take up to 3 chars to catch "min")
            for prefix in sorted(_FREQ_TO_PERIOD, key=len, reverse=True):
                if raw_freq.startswith(prefix):
                    sp = _FREQ_TO_PERIOD[prefix]
                    break

        if sp is not None and len(series) < 2 * sp:
            # Not enough data to identify seasonal pattern — drop it gracefully
            self._fell_back = True
            return (None, None)

        return (sp, self._seasonal if sp is not None else None)
