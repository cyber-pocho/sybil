"""
Abstract base for all forecasting models.

Keeping this thin on purpose: one fit, one predict, one name.
Every concrete model (Prophet, ARIMA, LSTM, …) implements these three
and plugs straight into the pipeline without any further ceremony.

The only shared logic here is _validate_is_fitted(), which guards predict()
calls before fit() has run — a runtime error that would otherwise surface as
a confusing AttributeError deep inside the model.
"""

from abc import ABC, abstractmethod

import pandas as pd

from forecasting.schemas import ForecastResult, ModelCard


class BaseForecaster(ABC):

    _fitted: bool = False   # flipped to True by subclasses after a successful fit()

    # ── contract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, series: pd.Series, **kwargs) -> None:
        """Train the model on `series`.

        Implementations must set self._fitted = True on success so that
        predict() can verify the model is ready.
        """

    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult:
        """Return point forecasts and prediction intervals for `horizon` steps ahead.

        Call self._validate_is_fitted() at the top of every implementation —
        it raises a clean RuntimeError if fit() was never called.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short stable identifier, e.g. 'prophet' or 'arima'.

        Used as ForecastResult.model_name and for logging/routing.
        """

    @property
    @abstractmethod
    def card(self) -> ModelCard:
        """Static metadata describing what this model is good for.

        Returned by the agent when it explains its model selection to the user.
        """

    # ── shared helpers ────────────────────────────────────────────────────────

    def _validate_is_fitted(self) -> None:
        # Prevents a cryptic AttributeError from a half-initialised model object
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet — call fit() before predict()."
            )
