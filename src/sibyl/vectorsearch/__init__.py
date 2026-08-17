"""
Semantic retrieval over past forecasts.

Three files, one per question:

    embeddings.py   how a piece of text becomes a vector
    search.py       how vectors get compared
    store.py        where they live and how they come back

Nothing here knows about HTTP or Celery. The API calls `store.recall`, the worker
calls `store.remember`, and both pass an embedder in — the same injection seam the
agent uses for its LLM, and for the same reason: the tests need a deterministic
stand-in, and CI must stay green with no model weights installed.
"""
