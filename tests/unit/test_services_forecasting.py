"""
Tests for the forecast service — the layer the API and the worker share.

It takes records and returns a dict, with no database and no queue anywhere near
it, which is the whole reason it exists: these run in milliseconds and need no
infrastructure at all.
"""

import numpy as np
import pandas as pd
import pytest

from sibyl.services.forecasting import run_forecast

RNG = np.random.default_rng(7)


def _records(n: int, freq: str = "D", start: str = "2023-01-01") -> list[dict]:
    t = np.arange(n)
    y = 100.0 + 0.1 * t + 6.0 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 1.5, n)
    dates = pd.date_range(start, periods=n, freq=freq)
    return [{"date": d.strftime("%Y-%m-%d"), "sales": v} for d, v in zip(dates, y)]


@pytest.fixture(scope="module")
def result() -> dict:
    return run_forecast(_records(200), target_column="sales", model_name="ets", horizon=10)


# ── the happy path returns everything a caller needs ──────────────────────────

def test_returns_diagnosis_summary_and_forecast(result):
    assert set(result) == {"diagnosis", "summary", "forecast"}


def test_forecast_has_one_point_per_step(result):
    assert len(result["forecast"]["point_forecast"]) == 10


def test_intervals_are_ordered(result):
    f = result["forecast"]
    assert all(
        lo <= pt <= hi
        for lo, pt, hi in zip(f["lower_95"], f["point_forecast"], f["upper_95"])
    )


def test_result_is_json_ready(result):
    """mode="json" is what lets this go straight into a JSON column or a response.

    Without it the timestamps are datetime objects and the first thing that tries
    to serialise the row fails, in the worker, long after the mistake was made.
    """
    import json

    json.dumps(result)   # must not raise
    assert isinstance(result["forecast"]["timestamps"][0], str)


def test_diagnosis_is_the_full_report(result):
    assert result["diagnosis"]["target_column"] == "sales"
    assert "profile" in result["diagnosis"]


# ── errors are the caller's to fix, and say so ────────────────────────────────

def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        run_forecast(_records(200), model_name="arima")


def test_series_below_the_sample_floor_is_rejected():
    # 60 points is above ETS's floor of 24 and below prophet's of 100.
    with pytest.raises(ValueError, match="needs at least 100") as exc:
        run_forecast(_records(60, freq="ME"), model_name="prophet")
    # The refusal must name the way out, not just the problem.
    assert "Models that fit: ets" in str(exc.value)


def test_a_model_that_fits_the_same_short_series_runs():
    out = run_forecast(_records(60, freq="ME"), model_name="ets", horizon=6)
    assert len(out["forecast"]["point_forecast"]) == 6


def test_input_without_a_time_axis_is_rejected():
    with pytest.raises(ValueError):
        run_forecast([{"sales": float(v)} for v in range(200)])


def test_unknown_target_column_is_rejected():
    with pytest.raises(ValueError):
        run_forecast(_records(200), target_column="nope")


# ── conformal intervals are available through the service too ─────────────────

def test_conformal_wrapping_is_honoured():
    out = run_forecast(_records(300), target_column="sales", model_name="ets", conformal=True)
    assert out["forecast"]["model_name"] == "conformal(ets)"


def test_conformal_doubles_the_effective_sample_floor():
    # ETS needs 24; conformal(ets) needs max(48, 50) = 50.
    with pytest.raises(ValueError, match="needs at least 50"):
        run_forecast(_records(40), model_name="ets", conformal=True)
