"""
Seasonality detection via FFT power spectrum + ACF confirmation.

Algorithm:
  1. Linearly detrend the series so trend power doesn't swamp oscillations.
  2. Compute the one-sided FFT power spectrum; each bin corresponds to a
     frequency (cycles/sample) whose reciprocal is the period in samples.
  3. For every integer candidate period, accept it only when the ACF at that
     lag also clears the Bartlett ±2/√n significance band — this weeds out
     FFT peaks that are spectral leakage rather than true periodicity.
  4. seasonality_strength = power in accepted periods / total detrended power,
     giving an interpretable 0–1 fraction of variance explained by seasonality.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from diagnosis.schemas import SeasonalityResult

# Maps the pandas freq alias to the number of observations per natural cycle.
# Used to set the ACF lag window to 2× the expected period.
_PERIOD_MAP: dict[str, int] = {
    "min": 60, "T": 60,          # minute data → 60-sample hour
    "H": 24,   "h": 24,          # hourly data  → 24-sample day
    "D": 7,                       # daily data   → 7-sample week
    "W": 52,                      # weekly data  → 52-sample year
    "M": 12,   "MS": 12, "ME": 12,
    "Q": 4,    "QS": 4,
    "A": 1,    "Y": 1,
}

_CI_FACTOR = 2.0    # Bartlett 95 % confidence band multiplier: ±2/√n
_MIN_OBS   = 16     # need at least a couple of full cycles for FFT to be meaningful


def detect_seasonality(series: pd.Series, freq: str) -> SeasonalityResult:
    s = series.dropna().values.astype(float)
    n = len(s)

    if n < _MIN_OBS:
        return SeasonalityResult(
            has_seasonality=False,
            detected_periods=[],
            seasonality_strength=0.0,
            dominant_period=None,
        )

    expected_period = _PERIOD_MAP.get(freq, 7)

    # ── 1. linear detrend ────────────────────────────────────────────────────
    t = np.arange(n)
    coeffs   = np.polyfit(t, s, 1)           # fit y = a*t + b
    detrended = s - np.polyval(coeffs, t)    # remove the fitted line

    # ── 2. FFT power spectrum ────────────────────────────────────────────────
    fft_power = np.abs(np.fft.rfft(detrended)) ** 2   # one-sided power
    freqs     = np.fft.rfftfreq(n)                    # cycles per sample, length n//2+1

    # Skip DC (index 0) — it's the mean, not a periodic component.
    # Also skip any bin whose period would be < 2 samples (below Nyquist).
    valid   = (freqs[1:] > 0) & ((1 / freqs[1:]) >= 2)
    periods = np.round(1 / freqs[1:][valid]).astype(int)   # integer periods
    power   = fft_power[1:][valid]

    total_power = fft_power[1:].sum()   # normaliser; excludes DC

    # ── 3. ACF confirmation ──────────────────────────────────────────────────
    max_lag = min(2 * expected_period, n // 2 - 1)
    acf_vals = acf(detrended, nlags=max_lag, fft=True)     # length max_lag+1
    ci       = _CI_FACTOR / np.sqrt(n)                      # significance threshold

    detected: list[int] = []
    detected_power: float = 0.0
    dominant_period: int | None = None
    dominant_power:  float = 0.0

    for p in sorted(set(periods)):
        if p > max_lag:
            # Period exceeds the ACF window — can't confirm it
            continue

        # Sum power from all FFT bins that round to this integer period
        p_power = float(power[periods == p].sum())

        # ACF at lag p must clear the significance threshold to count
        acf_at_p = float(acf_vals[p]) if p < len(acf_vals) else 0.0
        if acf_at_p <= ci:
            continue

        detected.append(int(p))
        detected_power += p_power

        if p_power > dominant_power:   # track the period with most power
            dominant_power  = p_power
            dominant_period = int(p)

    # ── 4. strength ──────────────────────────────────────────────────────────
    strength = float(np.clip(detected_power / total_power, 0.0, 1.0)) if total_power > 0 else 0.0

    return SeasonalityResult(
        has_seasonality=len(detected) > 0,
        detected_periods=sorted(detected),
        seasonality_strength=round(strength, 4),
        dominant_period=dominant_period,
    )
