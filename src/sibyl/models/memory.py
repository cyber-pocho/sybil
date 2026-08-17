"""
A remembered forecast: what the data looked like, and what was done about it.

One row per completed job. The text is the diagnosis digest — the same six lines
the API returns — and the vector is that text embedded. Search over those vectors
answers the question the job rows cannot: *have we seen a series like this one
before, and what did we run on it?*

The table is the source of truth and there is no index file. That is a deliberate
simplification: an index on disk is a cache, and a cache of a table this small
buys milliseconds while introducing the one bug class that is genuinely nasty
here — a stale index that returns confident answers about rows that no longer
exist. Reading the vectors back per query costs a scan the database is happy to
do until there are far more memories than a service like this accumulates.
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import JSON, DateTime, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from sibyl.db.base import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # Unique: re-indexing a job replaces its memory rather than adding a second
    # copy that would then occupy two of the caller's k results with one series.
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)

    # Exactly what was embedded, kept verbatim. Without it a stored vector is an
    # unlabelled point in 384-space and there is no way to check a bad result.
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # The answer a search is usually looking for: which forecaster was run.
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)

    # float32 bytes rather than a JSON array of floats: 4 bytes a dimension
    # instead of ~20, and no decimal round trip between what was computed and
    # what comes back. Portable across SQLite and Postgres, unlike pgvector.
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Which model produced `embedding`, and how long it is. Indexed because every
    # query filters on it: cosine similarity between two different embedding
    # models is not a weak signal, it is a meaningless number that still sorts.
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)

    # Horizon, target column, conformal flag — whatever makes a hit readable
    # without a second query. JSON for the same reason the job row uses it.
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    @property
    def vector(self) -> np.ndarray:
        """The stored bytes back as a (dim,) float32 array.

        `frombuffer` is a view over immutable bytes, so copy: numpy will refuse
        to write to it otherwise, and the caller stacks these into a matrix.
        """
        return np.frombuffer(self.embedding, dtype=np.float32).copy()

    def __repr__(self) -> str:
        return f"<Memory {self.id} job={self.job_id} model={self.model_name}>"
