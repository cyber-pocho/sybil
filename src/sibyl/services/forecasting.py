"""
Diagnose a series, then forecast it.

Deliberately pure: no database, no Celery, no HTTP. It takes records and returns
a dict, which means the same function backs the API, the worker, and the tests,
and the tests need no infrastructure to run it.

It is also where the sample floor is enforced for the non-agent path. The agent
enforces the same rule in its own tool, phrased for a model that can retry; here
the caller gets a ValueError, because an HTTP client cannot pick again.
"""

from typing import Any

import pandas as pd

from diagnosis.pipeline import run_full_diagnosis
from forecasting.registry import MODELS, build, models_that_fit


def run_forecast(
    records: list[dict[str, Any]],
    target_column: str | None = None,
    model_name: str = "prophet",
    horizon: int = 30,
    conformal: bool = False,
) -> dict[str, Any]:
    """Diagnose `records`, fit `model_name`, and forecast `horizon` steps.

    Args:
        records:       row-oriented rows, i.e. DataFrame.to_dict("records").
        target_column: the column to forecast; defaults to the first numeric one.
        model_name:    a key of forecasting.registry.MODELS.
        horizon:       steps ahead.
        conformal:     replace native intervals with split-conformal ones.

    Returns a JSON-ready dict: the diagnosis report, its six-line digest, and the
    forecast. `mode="json"` on the dumps is what makes the timestamps strings, so
    the result can go straight into a JSON column or a response body.

    Raises ValueError for anything the caller can fix — an unknown model, a series
    too short for it, or input the diagnosis pipeline rejects.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(MODELS)}.")

    df = pd.DataFrame(records)

    # run_full_diagnosis raises ValueError for input it cannot diagnose, which is
    # already the contract this function promises — let it through untouched.
    report = run_full_diagnosis(df, target_column=target_column)

    series = _series_from(df, report.target_column, report.profile.datetime_column)
    model = build(model_name, conformal=conformal)

    # Same floor the agent enforces, same reason: every ModelCard states a minimum
    # and no forecaster checks it. Name the models that do fit, so the message is
    # actionable rather than just a refusal.
    usable = int(series.notna().sum())
    floor = model.card.min_samples_required
    if usable < floor:
        fits = models_that_fit(usable)
        raise ValueError(
            f"{model.card.name} needs at least {floor} usable observations and this "
            f"series has {usable}. "
            + (f"Models that fit: {', '.join(fits)}." if fits else "No model fits.")
        )

    model.fit(series)
    result = model.predict(horizon)

    return {
        "diagnosis": report.model_dump(mode="json"),
        "summary": report.to_summary(),
        "forecast": result.model_dump(mode="json"),
    }


def _series_from(df: pd.DataFrame, target_column: str, datetime_column: str | None) -> pd.Series:
    """Build the DatetimeIndex-backed series the forecasters require.

    `datetime_column` comes from the diagnosis that already ran. Re-deriving it
    here would mean two heuristics that can disagree, and a series indexed by one
    column while the report describes another is a genuinely confusing bug.
    """
    if datetime_column is None:
        raise ValueError(
            "No datetime column found. Forecasting needs a time axis; include a "
            "column of parseable dates."
        )
    index = pd.to_datetime(df[datetime_column], errors="coerce")
    series = pd.Series(pd.to_numeric(df[target_column], errors="coerce").to_numpy(), index=index)
    return series.sort_index()
