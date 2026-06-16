"""
Anomaly detection for univariate time-series.

Two methods are offered and selected automatically:

  z-score         — fast, interpretable, correct when data is roughly Gaussian.
                    Flag points whose |z| > threshold (default 3σ).

  isolation forest — tree-based, makes no distributional assumption.
                    Works by isolating points with short average path lengths
                    in random binary trees; anomalies are easier to isolate.

"auto" mode runs Shapiro-Wilk on a 500-sample draw to test normality cheaply,
then picks z-score if Gaussian (p > 0.05) and isolation forest otherwise.
Shapiro-Wilk is capped at 500 samples because it loses power at n > 5000 and
the test itself becomes the bottleneck.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest

from diagnosis.schemas import AnomalyDetail, AnomalyResult

_SHAPIRO_MAX_SAMPLES = 500   # Shapiro-Wilk is O(n²); cap to keep it fast
_NORMALITY_ALPHA     = 0.05  # p-value threshold: above → treat as Gaussian


def zscore_anomalies(
    series: pd.Series,
    threshold: float = 3.0,
) -> list[AnomalyDetail]:
    """Flag points whose absolute z-score exceeds `threshold`."""
    s = series.dropna()
    if s.std() == 0:
        return []   # constant series: every z-score is 0, nothing is anomalous

    mu, sigma = s.mean(), s.std(ddof=1)
    zscores = (s - mu) / sigma   # standard z-score: distance from mean in σ units

    flagged = s[np.abs(zscores) > threshold]
    return [
        AnomalyDetail(
            index=int(idx),
            value=float(val),
            score=round(float(zscores.loc[idx]), 4),
        )
        for idx, val in flagged.items()
    ]


def isolation_forest_anomalies(
    series: pd.Series,
    contamination: float = 0.05,
) -> list[AnomalyDetail]:
    """Flag points classified as anomalies by an IsolationForest.

    The model returns a raw anomaly score in (-0.5, 0.5] where more negative
    means more anomalous; we store that directly as `score` for interpretability.
    """
    s = series.dropna()
    X = s.values.reshape(-1, 1)   # sklearn expects 2-D input

    clf = IsolationForest(contamination=contamination, random_state=0)
    labels = clf.fit_predict(X)           # +1 = inlier, -1 = anomaly
    scores = clf.decision_function(X)     # higher = more normal; anomalies < 0

    anomaly_mask = labels == -1
    return [
        AnomalyDetail(
            index=int(s.index[i]),
            value=float(s.iloc[i]),
            score=round(float(scores[i]), 4),
        )
        for i in np.where(anomaly_mask)[0]
    ]


def detect_anomalies(
    series: pd.Series,
    method: str = "auto",
) -> AnomalyResult:
    """
    Args:
        series: univariate time-series, may contain NaNs.
        method: "zscore", "isolation_forest", or "auto".
                "auto" chooses based on a Shapiro-Wilk normality test.
    """
    s = series.dropna()

    chosen = _pick_method(s) if method == "auto" else method

    if chosen == "zscore":
        details = zscore_anomalies(s)
    else:
        details = isolation_forest_anomalies(s)
        chosen = "isolation_forest"   # normalise in case caller passed a variant

    return AnomalyResult(
        anomaly_count=len(details),
        anomaly_indices=[d.index for d in details],
        method_used=chosen,
        details=details,
    )


# ── internal ──────────────────────────────────────────────────────────────────

def _pick_method(s: pd.Series) -> str:
    """Return "zscore" if the series looks Gaussian, "isolation_forest" otherwise."""
    sample = s if len(s) <= _SHAPIRO_MAX_SAMPLES else s.sample(_SHAPIRO_MAX_SAMPLES, random_state=0)
    _, p = scipy_stats.shapiro(sample)
    # High p-value → fail to reject normality → z-score is appropriate
    return "zscore" if p > _NORMALITY_ALPHA else "isolation_forest"
