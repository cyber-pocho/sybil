"""
Stationarity testing via ADF and KPSS.

Two complementary tests are used because each has opposite null hypotheses:

  ADF  — H0: unit root present (non-stationary).
          p < 0.05 → reject H0 → evidence of stationarity.

  KPSS — H0: series is level-stationary.
          p < 0.05 → reject H0 → evidence of non-stationarity.

Calling a series stationary requires both to agree:
  ADF p < 0.05  AND  KPSS p > 0.05.

When they disagree the series is in an ambiguous regime (e.g. trend-stationary
but not difference-stationary). recommended_differencing still applies the
practical fix used by ARIMA pipelines: try d=1, then d=2.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from diagnosis.schemas import StationarityResult

_MIN_OBS = 20   # below this both tests have unreliable asymptotic distributions


def test_stationarity(series: pd.Series) -> StationarityResult:
    s = series.dropna()

    if len(s) < _MIN_OBS:
        return StationarityResult(
            is_stationary=False,
            adf_pvalue=None,
            kpss_pvalue=None,
            recommended_differencing=0,
            note=f"series too short ({len(s)} obs, need ≥ {_MIN_OBS})",
        )

    if s.std() == 0:
        # Constant series: trivially stationary — no unit root, no stochastic trend.
        return StationarityResult(
            is_stationary=True,
            adf_pvalue=0.0,
            kpss_pvalue=1.0,
            recommended_differencing=0,
            note="constant series",
        )

    adf_p = _adf_pvalue(s)
    kpss_p = _kpss_pvalue(s)

    is_stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    d = _recommend_differencing(s, is_stationary)

    return StationarityResult(
        is_stationary=is_stationary,
        adf_pvalue=round(float(adf_p), 6),
        kpss_pvalue=round(float(kpss_p), 6),
        recommended_differencing=d,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _adf_pvalue(s: pd.Series) -> float:
    result = adfuller(s, autolag="AIC")
    return result[1]   # index 1 is always the p-value


def _kpss_pvalue(s: pd.Series) -> float:
    # statsmodels emits a UserWarning when the p-value is pinned at a table
    # boundary (0.01 or 0.1); suppress it — the boundary value is still usable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(s, regression="c", nlags="auto")
    return result[1]


def _recommend_differencing(s: pd.Series, already_stationary: bool) -> int:
    if already_stationary:
        return 0

    # First difference removes stochastic trends (random walks, I(1) processes).
    d1 = s.diff().dropna()
    if len(d1) >= _MIN_OBS and _adf_pvalue(d1) < 0.05 and _kpss_pvalue(d1) > 0.05:
        return 1

    # Second difference handles I(2) processes; rarely needed in practice.
    d2 = d1.diff().dropna()
    if len(d2) >= _MIN_OBS and _adf_pvalue(d2) < 0.05 and _kpss_pvalue(d2) > 0.05:
        return 2

    return 2   # conservative fallback — over-differencing is safer than under
