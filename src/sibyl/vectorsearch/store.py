"""
Writing memories down and getting them back.

`remember` is called by the worker after a job succeeds; `recall` is called by
the API. Both take an `Embedder` rather than building one, so the tests can pass
a three-dimensional stand-in and never load a model.

The retrieval is the obvious thing: pull every vector in the query's embedding
space out of the table, stack them, and take the top k by cosine. No index file,
no approximation, no staleness — see models/memory.py for why that trade is the
right one at this size.
"""

import uuid
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel
from sqlalchemy import delete, select

from sibyl.db.engine import session_scope
from sibyl.models.memory import Memory
from sibyl.vectorsearch.embeddings import Embedder
from sibyl.vectorsearch.search import cosine_top_k


class MemoryHit(BaseModel):
    """One retrieved memory, with how close it was."""

    job_id: str
    score: float          # cosine similarity in [-1, 1]; 1 is the same direction
    text: str             # the diagnosis digest that was embedded
    model_name: str       # the forecaster that was run on it — usually the answer
    meta: dict[str, Any]
    created_at: datetime


def job_document(result: dict[str, Any]) -> str:
    """The text to embed for a finished job: its diagnosis digest.

    Deliberately *only* the digest. Appending "model used: ets" would make the
    stored point drift toward the wording of the answer, and searching is meant
    to find series that look alike — the model that was chosen is a column on the
    row, so it comes back with the hit without polluting the geometry.
    """
    return result["summary"]


def remember(
    job_id: str,
    text: str,
    model_name: str,
    meta: dict[str, Any],
    embedder: Embedder,
) -> str:
    """Embed `text` and store it against `job_id`, replacing any earlier memory of it.

    Returns the memory id. Replacing rather than appending keeps a re-run job from
    occupying two of a caller's k results with the same series.
    """
    vector = np.asarray(embedder.encode([text])[0], dtype=np.float32)

    memory_id = str(uuid.uuid4())
    with session_scope() as session:
        session.execute(delete(Memory).where(Memory.job_id == job_id))
        session.add(Memory(
            id=memory_id,
            job_id=job_id,
            text=text,
            model_name=model_name,
            embedding=vector.tobytes(),
            embedding_model=embedder.name,
            dim=int(vector.shape[0]),
            meta=meta,
        ))
    return memory_id


def remember_job(job_id: str, params: dict[str, Any], result: dict[str, Any],
                 embedder: Embedder) -> str:
    """`remember`, given a finished job's stored params and result.

    Keeps the worker from having to know which fields of either make a memory —
    it hands over what the row already holds.
    """
    return remember(
        job_id=job_id,
        text=job_document(result),
        model_name=params.get("model_name", "unknown"),
        meta={
            "target_column": result["diagnosis"]["target_column"],
            "horizon": params.get("horizon"),
            "conformal": params.get("conformal", False),
            "frequency": result["diagnosis"]["profile"]["detected_frequency"],
            "row_count": result["diagnosis"]["profile"]["row_count"],
        },
        embedder=embedder,
    )


def recall(query: str, k: int, embedder: Embedder) -> list[MemoryHit]:
    """The `k` stored memories whose text is most similar to `query`.

    Only memories embedded by *this* embedder are considered. Mixing embedding
    spaces does not degrade the ranking gracefully; it produces numbers in the
    same [-1, 1] range that mean nothing at all, which is worse than returning
    fewer results.
    """
    q = np.asarray(embedder.encode([query])[0], dtype=np.float32)

    with session_scope() as session:
        rows = session.scalars(
            select(Memory).where(
                Memory.embedding_model == embedder.name,
                Memory.dim == int(q.shape[0]),
            )
        ).all()

        if not rows:
            return []

        matrix = np.stack([row.vector for row in rows])
        indices, scores = cosine_top_k(matrix, q, k)

        return [
            MemoryHit(
                job_id=rows[i].job_id,
                score=float(score),
                text=rows[i].text,
                model_name=rows[i].model_name,
                meta=rows[i].meta,
                created_at=rows[i].created_at,
            )
            for i, score in zip(indices, scores)
        ]
