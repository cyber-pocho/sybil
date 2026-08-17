"""
First live run of vector search against a real embedding model.

Every test in the suite injects a stub embedder, which is the only way the layer
runs in CI and is exactly why this script exists: the tests prove the search
maths, the store and the endpoint, and say nothing about whether
`SentenceTransformerEmbedder` produces a usable vector at all.

    pip install -e ".[db,vectorsearch]"
    python scripts/run_vectorsearch_live.py

It runs three real forecast jobs end to end — diagnosis, fit, predict, embed,
store — against a scratch SQLite file, then searches. No broker and no API: the
Celery task is a three-line wrapper over `run_forecast_job`, so calling the
function covers everything except message delivery.

Three fixtures that a good embedding should be able to tell apart, and a fourth
series used only as a query. The daily-with-a-weekly-cycle query should retrieve
the daily-with-a-weekly-cycle memory, not the monthly one. That is a weak claim
about the model and a strong claim about the wiring, which is the right split for
a script whose job is to prove the wiring.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RNG = np.random.default_rng(0)


def daily_weekly(n: int = 365) -> list[dict]:
    """Daily points with a strong weekly cycle — the shape prophet is built for."""
    t = np.arange(n)
    y = 100.0 + 0.1 * t + 8.0 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 1.5, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return [{"date": d.strftime("%Y-%m-%d"), "sales": v} for d, v in zip(dates, y)]


def monthly_trend(n: int = 60) -> list[dict]:
    """Monthly points, trend only — no cycle, and below prophet's floor of 100."""
    t = np.arange(n)
    y = 500.0 + 2.5 * t + RNG.normal(0, 8.0, n)
    dates = pd.date_range("2019-01-31", periods=n, freq="ME")
    return [{"date": d.strftime("%Y-%m-%d"), "sales": v} for d, v in zip(dates, y)]


def hourly_noisy(n: int = 480) -> list[dict]:
    """Hourly points with a daily cycle and three injected spikes."""
    t = np.arange(n)
    y = 50.0 + 5.0 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 1.0, n)
    y[[97, 250, 401]] += 40.0
    dates = pd.date_range("2024-03-01", periods=n, freq="h")
    return [{"date": d.strftime("%Y-%m-%d %H:%M:%S"), "sales": v} for d, v in zip(dates, y)]


def run_job(records: list[dict], model_name: str, horizon: int, label: str) -> str:
    """Write a pending job the way the API does, then run it the way the worker does."""
    import uuid

    from sibyl.db.engine import session_scope
    from sibyl.models.job import Job, JobStatus
    from sibyl.tasks.forecast import run_forecast_job

    job_id = str(uuid.uuid4())
    with session_scope() as session:
        session.add(Job(id=job_id, status=JobStatus.pending, params={
            "records": records, "target_column": "sales",
            "model_name": model_name, "horizon": horizon, "conformal": False,
        }))

    status = run_forecast_job(job_id)
    print(f"  {label:<34} {model_name:<8} → {status}  ({job_id[:8]})")
    return job_id


if __name__ == "__main__":
    # Default to a scratch database rather than a developer's real sibyl.db,
    # which would otherwise end up with three synthetic memories in it. An
    # explicit DATABASE_URL wins, which is how this gets pointed at a real
    # Postgres — `LargeBinary` is BLOB on SQLite and BYTEA there, and a vector
    # that round-trips through one has not been shown to round-trip the other.
    #
    # Set before anything imports the engine: the URL is read on first use.
    if not os.environ.get("DATABASE_URL"):
        scratch = tempfile.mkdtemp(prefix="sibyl-vectorsearch-")
        os.environ["DATABASE_URL"] = f"sqlite:///{scratch}/live.db"

    from sibyl.db.base import Base
    from sibyl.db.engine import get_engine
    from sibyl.models import job, memory  # noqa: F401  (registers both on the metadata)
    from sibyl.vectorsearch.embeddings import build_embedder
    from sibyl.vectorsearch.store import recall

    Base.metadata.create_all(get_engine())

    embedder = build_embedder()
    print(f"embedding model: {embedder.name}")
    print(f"database:        {os.environ['DATABASE_URL']}\n")

    # One embed up front, so a missing package or a bad model id fails here with
    # a clear message rather than inside the worker's best-effort try/except,
    # which is designed to swallow exactly this and log it as a warning.
    probe = embedder.encode(["a daily series with a weekly cycle"])
    print(f"probe: shape {probe.shape}, dtype {probe.dtype}, "
          f"norm {np.linalg.norm(probe[0]):.3f}\n")

    print("running three jobs end to end (diagnose → fit → predict → embed → store):")
    run_job(daily_weekly(), "prophet", 30, "365 daily, weekly cycle")
    run_job(monthly_trend(), "ets", 12, "60 monthly, trend only")
    run_job(hourly_noisy(), "ets", 24, "480 hourly, daily cycle + spikes")

    # The query is a *fourth* series, never indexed. Searching with one of the
    # three would only prove that a vector equals itself.
    from diagnosis.pipeline import run_full_diagnosis

    query_records = daily_weekly(n=400)
    digest = run_full_diagnosis(pd.DataFrame(query_records), target_column="sales").to_summary()

    print(f"\nquery — a fourth, unindexed daily series:\n{digest}\n")
    print("hits:")
    for rank, hit in enumerate(recall(digest, k=3, embedder=embedder), start=1):
        first_line = hit.text.splitlines()[0]
        print(f"  {rank}. {hit.score:+.3f}  {hit.model_name:<8} {hit.meta['frequency']:<9} "
              f"{first_line}")

    print(
        "\nA correct result puts the daily memory first. That is a weak claim about the "
        "\nembedding model and a strong one about the wiring, which is what this script is for."
    )
