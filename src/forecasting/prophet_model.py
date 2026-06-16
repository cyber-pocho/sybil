"""
Prophet forecaster wrapping Meta's additive decomposition model.

Prophet decomposes a time series as:
    y(t) = trend(t) + seasonality(t) + holidays(t) + noise

The trend is a piecewise-linear (or logistic) curve with automatic changepoint
detection. Seasonalities are modelled as Fourier series. This makes it robust
to missing data and outlier observations without any preprocessing.

Interval width note: Prophet's predict() takes quantiles from the posterior
predictive at the stored self.interval_width value. Changing that attribute
between two predict() calls on the same fitted model is safe — the point
forecast (yhat) is unchanged; only the quantile tails shift.
"""

import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

import pandas as pd
from prophet import Prophet

from forecasting.base import BaseForecaster
from forecasting.schemas import ForecastResult, ModelCard

# Prophet delegates to cmdstanpy (a Stan backend) which logs aggressively.
# Silence both at import time so nothing leaks into application logs.
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetForecaster(BaseForecaster):

    def __init__(
        self,
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality:  str | bool = "auto",
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        self._yearly_seasonality     = yearly_seasonality
        self._weekly_seasonality     = weekly_seasonality
        self._daily_seasonality      = daily_seasonality
        self._changepoint_prior_scale = changepoint_prior_scale
        self._model: Prophet | None  = None
        self._fit_time: float        = 0.0

    # ── BaseForecaster contract ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "prophet"

    @property
    def card(self) -> ModelCard:
        return ModelCard(
            name="prophet",
            description=(
                "Meta's additive decomposition model. Fits piecewise-linear trend, "
                "Fourier-basis seasonalities, and optional holiday effects. "
                "Tolerates gaps and outliers without preprocessing."
            ),
            best_for="Series with strong trend and clear weekly/yearly seasonality over ≥2 full seasonal cycles.",
            handles_missing=True,
            min_samples_required=100,   # need enough history for changepoint detection to be meaningful
            supports_uncertainty="native",
        )

    def fit(self, series: pd.Series, **kwargs: Any) -> None:
        """Convert series → Prophet DataFrame, then fit.

        Args:
            series: must carry a DatetimeIndex; Prophet needs real calendar dates
                    to decompose weekly and yearly seasonality correctly.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(
                "ProphetForecaster requires a DatetimeIndex. "
                "Pass series.set_index(<date_column>) before calling fit()."
            )

        # Prophet's column contract: 'ds' = datestamp, 'y' = observed value
        df = pd.DataFrame({"ds": series.index, "y": series.values})
        df = df.dropna()   # NaN in 'y' causes Stan errors; gaps in 'ds' are fine

        self._model = Prophet(
            yearly_seasonality=self._yearly_seasonality,
            weekly_seasonality=self._weekly_seasonality,
            daily_seasonality=self._daily_seasonality,
            changepoint_prior_scale=self._changepoint_prior_scale,
        )

        # redirect_stdout/stderr catches any residual print() calls from Prophet
        # that slip past the logging configuration above
        t0 = time.perf_counter()
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self._model.fit(df, **kwargs)
        self._fit_time = time.perf_counter() - t0

        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        """Forecast `horizon` steps ahead with 80 % and 95 % prediction intervals.

        Prophet produces intervals at a single width per predict() call, so we
        call predict() twice — once per interval level — on the same future
        dataframe. The point forecast (yhat) is identical in both calls.
        """
        self._validate_is_fitted()

        future = self._model.make_future_dataframe(periods=horizon)

        # 80 % interval — the tails most useful for operational planning
        self._model.interval_width = 0.80
        fc80 = self._model.predict(future).tail(horizon)

        # 95 % interval — standard statistical reporting convention
        self._model.interval_width = 0.95
        fc95 = self._model.predict(future).tail(horizon)

        return ForecastResult(
            timestamps=fc80["ds"].tolist(),
            point_forecast=fc80["yhat"].tolist(),
            lower_80=fc80["yhat_lower"].tolist(),
            upper_80=fc80["yhat_upper"].tolist(),
            lower_95=fc95["yhat_lower"].tolist(),
            upper_95=fc95["yhat_upper"].tolist(),
            model_name=self.name,
            fit_time_seconds=round(self._fit_time, 3),
            metadata={
                "changepoint_prior_scale": self._changepoint_prior_scale,
                "yearly_seasonality": self._yearly_seasonality,
                "weekly_seasonality": self._weekly_seasonality,
                "daily_seasonality":  self._daily_seasonality,
            },
        )
