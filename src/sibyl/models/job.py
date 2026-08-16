"""
The one persisted thing: a forecast job.

A job is a request that outlived the HTTP call that made it. The API writes it
`pending` and returns an id; the worker moves it to `running`, then `done` with a
result or `failed` with a message. Nothing else in the row is interesting — the
point is that a client can come back later and still find out what happened.

`params` carries the caller's raw records, because the worker has to fit on them
and the queue only moves an id. That makes a row roughly as large as the series
posted, which is fine at this scale and is the first thing to revisit if anyone
posts a million points: the fix is object storage plus a key, not a bigger column.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from sibyl.db.base import Base


class JobStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


def _now() -> datetime:
    return datetime.now(UTC)


class Job(Base):
    __tablename__ = "jobs"

    # A UUID string rather than a serial: the API hands it to a caller
    # immediately, so it cannot depend on what the database assigns on insert.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    status: Mapped[str] = mapped_column(
        String(16), default=JobStatus.pending, nullable=False, index=True
    )

    # JSON rather than columns-per-field: these are the request and the response,
    # and pinning either into a schema would mean a migration every time the
    # forecast payload gains a field.
    params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Text, not String: this is a caught exception message, and truncating the
    # one field that explains a failure would be a poor trade for a few bytes.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Job {self.id} {self.status}>"
