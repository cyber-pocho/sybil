"""
Text in, vectors out.

One implementation ships — a sentence-transformers model running locally — and it
sits behind a Protocol so tests can inject something deterministic. That is the
same seam `run_agent(..., llm=...)` uses, and it exists for the same reason: a
real model is slow, non-deterministic, and 90 MB of weights CI has no business
downloading.

Unlike the LLM provider registry next door, this one *does* have a default. The
asymmetry is deliberate and it is about consequences, not consistency: picking a
default chat provider fires a request at somebody's paid endpoint that the caller
never chose, while picking a default embedding model downloads open weights and
runs them on the local CPU. Nobody gets billed for guessing wrong here.

What is not defaulted is which model a *stored* vector came from. Two embedding
models produce two incompatible geometries, and cosine similarity between them is
not small — it is meaningless, and it looks exactly like a real score. So every
memory records the model that produced it and `store.recall` compares only within
one of those spaces.
"""

import os
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

# Small, fast, and the usual default for sentence similarity: 384 dimensions,
# ~90 MB, and it runs on CPU in milliseconds. Downloaded on first use.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Everything the rest of the codebase asks of an embedding model.

    `name` identifies the vector space, not the object — it is written to every
    row and used to keep incomparable vectors apart.
    """

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed `texts`, returning one row per text: shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    """A local sentence-transformers model, loaded on first use.

    Loading is lazy because constructing this happens on every worker job and
    every /search request, while the 90 MB of weights only need to arrive once —
    and because importing torch at module import time would make `import
    sibyl.api.app` cost several seconds for a service that may never embed
    anything.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = resolve_embedding_model(name)
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed, which vector search needs to "
                    'turn text into vectors. Install it with: pip install -e ".[vectorsearch]"'
                ) from exc
            self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        # float32 because that is what gets written to the database, and doing the
        # cast here means the stored bytes and the in-memory matrix never differ.
        vectors = self._load().encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=4)
def _cached_embedder(name: str) -> Embedder:
    """One instance per model name, so the weights load once per process.

    The first live run is what showed this was needed. Both callers build an
    embedder per unit of work — the worker per job, the endpoint per request —
    and every fresh instance reloads 90 MB of weights, so a three-job run paid
    for the model three times. The instance holds nothing but the loaded model,
    so sharing it is safe; `maxsize=4` bounds the memory if several names are
    ever in play at once.
    """
    return SentenceTransformerEmbedder(name)


def resolve_embedding_model(name: str | None = None) -> str:
    """The model id from the argument, else SIBYL_EMBEDDING_MODEL, else the default."""
    return name or os.environ.get("SIBYL_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def build_embedder(name: str | None = None) -> Embedder:
    """The embedder for `name`, or for whatever the environment selects.

    A factory for a single implementation, for the same reason `build_llm`
    exists: it is the one place the environment is read, so callers take an
    `Embedder` and nothing else in the codebase mentions an env var.

    The name is resolved *before* the cache rather than inside the constructor.
    Caching on the raw `None` would key every environment-selected model to the
    same entry, so changing SIBYL_EMBEDDING_MODEL would keep handing back the
    previous model — silently, and with results that still look like results.
    """
    return _cached_embedder(resolve_embedding_model(name))
