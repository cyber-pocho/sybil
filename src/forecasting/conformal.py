"""
Distribution-free conformal prediction intervals for any BaseForecaster.

Split-conformal prediction (Papadopoulos 2002, Angelopoulos & Bates 2021)
gives a finite-sample, assumption-free coverage guarantee:

    P(y_{n+1} ∈ [ŷ - q, ŷ + q]) ≥ 1 - α

where q is the ⌈(1-α)(n_cal+1)⌉ / n_cal quantile of the calibration
nonconformity scores. No distributional assumption is needed — only that
calibration and test residuals are exchangeable (approximately true for
stationary series).

Fit pipeline:
  1. Split series: train = first 70%, cal = last 30%
  2. fit base_forecaster on train
  3. Forecast calibration period; compute scores = |y_actual - ŷ|
  4. Refit base_forecaster on full series so predict() starts from the right point
  5. Store sorted scores; predict() applies the conformal quantile to the base
     forecaster's point predictions.
"""

import math

import numpy as np
import pandas as pd

from forecasting.base import BaseForecaster
from forecasting.schemas import ForecastResult, ModelCard

# ── main wrapper ─────────────────────────────────────────────────────────────

class ConformalWrapper(BaseForecaster):
    """
    Replaces a BaseForecaster's native prediction intervals with conformal ones.
    The point forecast is unchanged; only the interval half-widths come from the
    conformal quantile of calibration residuals.
    """

    def __init__(self, base_forecaster: BaseForecaster) -> None:
        self._base = base_forecaster
        self._scores: np.ndarray | None = None   # sorted nonconformity scores (ascending)
        self._n_cal: int = 0

    @property
    def name(self) -> str:
        return f"conformal({self._base.name})"

    @property
    def card(self) -> ModelCard:
        bc = self._base.card
        return ModelCard(
            name=self.name,
            description=(
                f"Split-conformal wrapper around {bc.name}. "
                "Replaces model-native intervals with finite-sample, "
                "distribution-free conformal ones."
            ),
            best_for=bc.best_for,
            handles_missing=bc.handles_missing,
            min_samples_required=max(bc.min_samples_required * 2, 50),
            supports_uncertainty="conformal",
        )

    def fit(self, series: pd.Series, **kwargs) -> None:
        n = len(series)
        # Integer arithmetic, not int(0.7 * n): 0.7 is not exactly representable,
        # so 0.7 * 350 == 244.99999999999997 and int() would silently truncate to
        # 244, giving a 106-point calibration set where 105 was intended.
        n_train = 7 * n // 10

        train = series.iloc[:n_train]
        cal   = series.iloc[n_train:]
        self._n_cal = len(cal)

        # ── step 1: fit on train, predict calibration period ─────────────────
        self._base.fit(train, **kwargs)
        cal_pred = self._base.predict(len(cal))

        # Nonconformity score: absolute residual at each calibration step.
        # Larger score → point is harder to predict → wider interval needed.
        scores = np.abs(cal.values - np.array(cal_pred.point_forecast))
        self._scores = np.sort(scores)   # ascending; quantile lookup is O(1)

        # ── step 2: refit on full series ──────────────────────────────────────
        # Without this, predict() would forecast from the 70% mark, not the end.
        self._base.fit(series, **kwargs)
        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        self._validate_is_fitted()

        base_result = self._base.predict(horizon)
        yhat = np.array(base_result.point_forecast)

        # conformal half-widths; same value applied to every forecast step
        q80 = _conformal_quantile(self._scores, self._n_cal, alpha=0.20)
        q95 = _conformal_quantile(self._scores, self._n_cal, alpha=0.05)

        return ForecastResult(
            timestamps=base_result.timestamps,
            point_forecast=yhat.tolist(),
            lower_80=(yhat - q80).tolist(),
            upper_80=(yhat + q80).tolist(),
            lower_95=(yhat - q95).tolist(),
            upper_95=(yhat + q95).tolist(),
            model_name=self.name,
            fit_time_seconds=base_result.fit_time_seconds,
            metadata={
                **base_result.metadata,
                "n_calibration": self._n_cal,
                "q80": round(float(q80), 4),
                "q95": round(float(q95), 4),
            },
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _conformal_quantile(scores: np.ndarray, n: int, alpha: float) -> float:
    """
    Exact finite-sample conformal quantile (Theorem 1, Angelopoulos & Bates 2021).

    q_{α} = scores[ ⌈(1-α)(n+1)⌉ - 1 ]   (0-indexed into sorted scores)

    When ⌈(1-α)(n+1)⌉ > n, the calibration set is too small to bound the
    interval at level α — return inf so predict() produces infinite-width
    intervals rather than silently under-covering.
    """
    level = math.ceil((1 - alpha) * (n + 1))   # 1-indexed position
    if level > n:
        return float("inf")     # need more calibration data
    return float(scores[level - 1])             # shift to 0-indexed
