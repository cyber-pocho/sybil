"""
Tests for the job row, the worker that fills it in, and the migration behind it.

The worker path is exercised by calling `run_forecast_job` directly. That is the
whole function the Celery task wraps, so everything except message delivery is
covered — and message delivery is Celery's to get right, not ours to assert.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sqlalchemy", reason="persistence requires the [db] extra")

from sibyl.db.engine import session_scope  # noqa: E402
from sibyl.models.job import Job, JobStatus  # noqa: E402
from sibyl.tasks.forecast import run_forecast_job  # noqa: E402

RNG = np.random.default_rng(3)


def _records(n: int, freq: str = "D") -> list[dict]:
    t = np.arange(n)
    y = 100.0 + 0.1 * t + 6.0 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 1.5, n)
    dates = pd.date_range("2023-01-01", periods=n, freq=freq)
    return [{"date": d.strftime("%Y-%m-%d"), "sales": v} for d, v in zip(dates, y)]


def _queue(params: dict) -> str:
    """Write a pending job the way the API does, and return its id."""
    job_id = "job-under-test"
    with session_scope() as session:
        session.add(Job(id=job_id, status=JobStatus.pending, params=params))
    return job_id


def _fetch(job_id: str) -> Job:
    with session_scope() as session:
        return session.get(Job, job_id)


# ── a job that succeeds ───────────────────────────────────────────────────────

@pytest.fixture
def finished(db) -> Job:
    job_id = _queue({
        "records": _records(200), "target_column": "sales",
        "model_name": "ets", "horizon": 7, "conformal": False,
    })
    run_forecast_job(job_id)
    return _fetch(job_id)


def test_status_is_done(finished):
    assert finished.status == JobStatus.done


def test_result_is_stored(finished):
    assert len(finished.result["forecast"]["point_forecast"]) == 7


def test_no_error_recorded(finished):
    assert finished.error is None


def test_timestamps_are_recorded(finished):
    assert finished.started_at is not None
    assert finished.finished_at is not None


# ── a job that fails is a recorded outcome, not a crash ───────────────────────

@pytest.fixture
def failed(db) -> Job:
    # 60 points against prophet's floor of 100 — the service raises, and the
    # worker's job is to write that down rather than let it escape.
    job_id = _queue({
        "records": _records(60, freq="ME"), "target_column": "sales",
        "model_name": "prophet", "horizon": 7, "conformal": False,
    })
    run_forecast_job(job_id)
    return _fetch(job_id)


def test_failed_status_is_recorded(failed):
    assert failed.status == JobStatus.failed


def test_failure_message_is_useful(failed):
    assert "needs at least 100" in failed.error


def test_failed_job_has_no_result(failed):
    assert failed.result is None


def test_failed_job_still_finishes(failed):
    # The row must not be left in `running` — a caller polling it would hang forever.
    assert failed.finished_at is not None


def test_worker_does_not_raise_on_a_failed_forecast(db):
    job_id = _queue({
        "records": _records(60, freq="ME"), "target_column": "sales",
        "model_name": "prophet", "horizon": 7, "conformal": False,
    })
    assert run_forecast_job(job_id) == JobStatus.failed


# ── a finished job is remembered, but never at the cost of the job ────────────

@pytest.fixture
def indexed(db, stub_embedder, monkeypatch) -> Job:
    """Run a job with the stub embedder standing in for the real one.

    `_remember` imports build_embedder inside the function, so patching the
    module it imports from is what reaches it — and that late import is itself
    the reason a worker without [vectorsearch] installed still runs jobs.
    """
    from sibyl.vectorsearch import embeddings

    monkeypatch.setattr(embeddings, "build_embedder", lambda *a, **k: stub_embedder)
    job_id = _queue({
        "records": _records(200), "target_column": "sales",
        "model_name": "ets", "horizon": 7, "conformal": False,
    })
    run_forecast_job(job_id)
    return _fetch(job_id)


def test_a_successful_job_is_indexed(indexed, stub_embedder):
    from sibyl.vectorsearch.store import recall

    hits = recall(indexed.result["summary"], k=1, embedder=stub_embedder)
    assert [h.job_id for h in hits] == [indexed.id]


def test_a_failed_job_is_not_indexed(db, stub_embedder, monkeypatch):
    from sibyl.vectorsearch import embeddings
    from sibyl.vectorsearch.store import recall

    monkeypatch.setattr(embeddings, "build_embedder", lambda *a, **k: stub_embedder)
    job_id = _queue({
        "records": _records(60, freq="ME"), "target_column": "sales",
        "model_name": "prophet", "horizon": 7, "conformal": False,
    })
    run_forecast_job(job_id)

    # There is no digest to remember and nothing to learn from: the forecast
    # never ran, so there is no answer to attach to the diagnosis.
    assert recall("anything", k=5, embedder=stub_embedder) == []


def test_an_indexing_failure_does_not_fail_the_job(db, monkeypatch):
    """The forecast already succeeded. A missing embedding model cannot undo that."""
    from sibyl.vectorsearch import embeddings

    def _explode(*args, **kwargs):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(embeddings, "build_embedder", _explode)
    job_id = _queue({
        "records": _records(200), "target_column": "sales",
        "model_name": "ets", "horizon": 7, "conformal": False,
    })

    assert run_forecast_job(job_id) == JobStatus.done
    assert _fetch(job_id).result is not None


# ── a job that does not exist is a different kind of problem ──────────────────

def test_missing_job_raises(db):
    # The queue and the database disagreeing is not something a retry fixes.
    with pytest.raises(LookupError):
        run_forecast_job("no-such-job")


# ── the migration and the models must not drift apart ─────────────────────────

def test_migration_matches_the_models(tmp_path, monkeypatch):
    """Run the real migration, then diff the schema against Base.metadata.

    Tables are created with create_all everywhere else in the suite, which would
    happily pass while alembic/versions was stale. This is the one test that
    would fail if someone added a column and forgot the migration.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.command import upgrade
    from alembic.config import Config
    from alembic.migration import MigrationContext

    from sibyl.db.base import Base
    from sibyl.db.engine import get_engine, reset_engine
    from sibyl.models import job, memory  # noqa: F401

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'migrated.db'}")
    reset_engine()

    config = Config("alembic.ini")
    upgrade(config, "head")

    with get_engine().connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    reset_engine()
    assert diff == [], f"models and migrations disagree: {diff}"
