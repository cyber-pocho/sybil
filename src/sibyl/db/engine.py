"""
Engine and session factory.

Sync SQLAlchemy on purpose. The Celery worker that does almost all of the writing
is itself sync, and an async engine there buys nothing but an asyncio bridge in
the one place a failure is hardest to see. The `asyncpg` dependency in the [db]
extra is unused for now; it stays for whenever a read path actually needs it.

The engine is built on first use rather than at import. Tests point DATABASE_URL
at a scratch file and call `reset_engine()`; if the engine were created at import
time, that would only work when the import order happened to cooperate.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

# SQLite by default so a fresh clone runs with no infrastructure. docker-compose
# overrides this with the postgres service.
DEFAULT_DATABASE_URL = "sqlite:///./sibyl.db"

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = database_url()
        # SQLite refuses a connection reused across threads unless told otherwise,
        # and FastAPI's threadpool does exactly that. Postgres needs no such help.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transaction that commits on success and rolls back on any exception.

    `expire_on_commit=False` above matters here: without it, attributes read after
    the commit would trigger a refresh against a closed session.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine so the next call re-reads DATABASE_URL. For tests."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
