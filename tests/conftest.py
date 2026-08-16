"""
Shared fixtures.

The database fixtures point DATABASE_URL at a per-test SQLite file and reset the
cached engine around it, so tests never touch a developer's real database and
never leak state into each other. Imports are inside the fixture bodies: these
are opt-in, and importing SQLAlchemy at collection time would break the suite for
anyone who installed the project without the [db] extra.
"""

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A scratch SQLite database with the schema created, isolated per test."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")

    from sibyl.db.base import Base
    from sibyl.db.engine import get_engine, reset_engine
    from sibyl.models import job  # noqa: F401  (registers Job on Base.metadata)

    reset_engine()
    Base.metadata.create_all(get_engine())
    yield
    reset_engine()


@pytest.fixture
def no_broker(monkeypatch):
    """Record dispatches instead of sending them to Redis.

    The API's contract is that it writes a job and hands it to the queue; whether
    Celery then delivers the message is Celery's problem, not something these
    tests can or should assert. So capture the call and check the id.
    """
    dispatched: list[str] = []

    from sibyl.api import app as app_module

    class _Recorder:
        @staticmethod
        def delay(job_id: str) -> None:
            dispatched.append(job_id)

    monkeypatch.setattr(app_module, "forecast_task", _Recorder)
    return dispatched
