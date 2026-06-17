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

try:
    from mapie.regression import MapieRegressor
    from sklearn.base import BaseEstimator, RegressorMixin
    _MAPIE_AVAILABLE = True
except ImportError:
    _MAPIE_AVAILABLE = False


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
        n       = len(series)
        n_train = int(0.7 * n)

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


# ── MAPIE-backed wrapper (requires pip install mapie) ────────────────────────

class MAPIEWrapper(BaseForecaster):
    """
    Conformal wrapper that delegates quantile calibration to MAPIE.

    Uses MapieRegressor with cv='prefit' and a lightweight sklearn stub that
    replays the base forecaster's point predictions. Compared to ConformalWrapper,
    MAPIE's API unlocks Jackknife+ and cross-conformal variants; here we use the
    standard split-conformal (same coverage math, validated MAPIE implementation).

    y_pis output shape from MAPIE ≥ 0.6: (n_samples, 2, n_alpha)
      axis-1 index 0 = lower bound, index 1 = upper bound.

    Raises ImportError at instantiation if MAPIE is not installed.
    """

    def __init__(self, base_forecaster: BaseForecaster) -> None:
        if not _MAPIE_AVAILABLE:
            raise ImportError("MAPIEWrapper requires MAPIE: pip install mapie")
        self._base    = base_forecaster
        self._mapie80 = None   # MapieRegressor calibrated at alpha=0.20
        self._mapie95 = None   # MapieRegressor calibrated at alpha=0.05
        self._n_cal: int = 0

    @property
    def name(self) -> str:
        return f"mapie({self._base.name})"

    @property
    def card(self) -> ModelCard:
        bc = self._base.card
        return ModelCard(
            name=self.name,
            description=(
                f"MAPIE-calibrated conformal wrapper around {bc.name}. "
                "Exposes Jackknife+ and cross-conformal variants via MAPIE's API."
            ),
            best_for=bc.best_for,
            handles_missing=bc.handles_missing,
            min_samples_required=max(bc.min_samples_required * 2, 50),
            supports_uncertainty="conformal",
        )

    def fit(self, series: pd.Series, **kwargs) -> None:
        n       = len(series)
        n_train = int(0.7 * n)

        train = series.iloc[:n_train]
        cal   = series.iloc[n_train:]
        self._n_cal = len(cal)

        # Fit base and collect calibration predictions
        self._base.fit(train, **kwargs)
        cal_pred    = self._base.predict(len(cal))
        cal_preds   = np.array(cal_pred.point_forecast)
        cal_actuals = cal.values.astype(float)

        # MAPIE's prefit mode needs (X, y) even though our stub ignores X.
        # We use the time index as X so MAPIE can validate array shapes.
        X_cal = np.arange(len(cal)).reshape(-1, 1)

        # Two separate MapieRegressors, one per alpha, so predict() can return
        # both interval widths without re-running calibration.
        for alpha, attr in [(0.20, "_mapie80"), (0.05, "_mapie95")]:
            stub  = _PrefitStub(cal_preds)            # gives calibration preds during fit()
            mapie = MapieRegressor(estimator=stub, cv="prefit")
            mapie.fit(X_cal, cal_actuals)             # MAPIE computes residuals here
            setattr(self, attr, mapie)

        # Refit base forecaster on full series
        self._base.fit(series, **kwargs)
        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        self._validate_is_fitted()

        base_result = self._base.predict(horizon)
        yhat        = np.array(base_result.point_forecast)

        # X_test is a dummy feature matrix; the stub ignores it and returns yhat
        X_test = np.arange(self._n_cal, self._n_cal + horizon).reshape(-1, 1)

        # Update stub predictions before MAPIE calls stub.predict(X_test) internally
        self._mapie80.estimator_.preds_ = yhat
        self._mapie95.estimator_.preds_ = yhat

        # y_pis shape: (horizon, 2, 1) — axis 1: [lower, upper]; axis 2: single alpha
        _, pi80 = self._mapie80.predict(X_test, alpha=0.20)
        _, pi95 = self._mapie95.predict(X_test, alpha=0.05)

        return ForecastResult(
            timestamps=base_result.timestamps,
            point_forecast=yhat.tolist(),
            lower_80=pi80[:, 0, 0].tolist(),
            upper_80=pi80[:, 1, 0].tolist(),
            lower_95=pi95[:, 0, 0].tolist(),
            upper_95=pi95[:, 1, 0].tolist(),
            model_name=self.name,
            fit_time_seconds=base_result.fit_time_seconds,
            metadata={**base_result.metadata, "n_calibration": self._n_cal},
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


class _PrefitStub:
    """
    Minimal sklearn-compatible stub that returns precomputed predictions.

    Used in MAPIEWrapper to put MapieRegressor in 'prefit' mode:
    - During mapie.fit(): returns calibration predictions so MAPIE can
      compute residuals without actually training anything.
    - During mapie.predict(): the caller updates preds_ to the test
      predictions before calling predict(), so MAPIE applies the cached
      conformal quantile to the right base predictions.
    """
    def __init__(self, preds: np.ndarray) -> None:
        self.preds_ = preds    # trailing _ follows sklearn's fitted-attribute convention

    def fit(self, X, y):
        return self            # no-op: model is externally prefit

    def predict(self, X) -> np.ndarray:
        return self.preds_     # caller is responsible for setting this before predict()

    # sklearn estimator protocol — needed for MapieRegressor's internal checks
    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params) -> "_PrefitStub":
        return self
