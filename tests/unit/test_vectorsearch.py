"""
Tests for the search maths, and for the store that keeps vectors alive.

The maths half needs nothing but numpy and asserts the actual numbers — cosine
has known answers for identical, orthogonal and opposite vectors, so there is no
excuse for asserting "is roughly sorted" instead.

The store half runs against a scratch SQLite database and the WordCountEmbedder
from conftest. No model weights are downloaded here, which is also what makes
these tests run in CI: [vectorsearch] is not installed there.
"""

import numpy as np
import pytest

from sibyl.vectorsearch.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    build_embedder,
    resolve_embedding_model,
)
from sibyl.vectorsearch.search import cosine_top_k, unit

pytest.importorskip("sqlalchemy", reason="the memory store requires the [db] extra")

from sibyl.vectorsearch.store import job_document, recall, remember, remember_job  # noqa: E402

# ── choosing an embedding model ───────────────────────────────────────────────
# None of these load weights: SentenceTransformerEmbedder defers that to the
# first encode, which is what makes the selection logic testable on its own.


def test_the_argument_wins(monkeypatch):
    monkeypatch.setenv("SIBYL_EMBEDDING_MODEL", "from-the-environment")
    assert resolve_embedding_model("explicit") == "explicit"


def test_the_environment_wins_over_the_default(monkeypatch):
    monkeypatch.setenv("SIBYL_EMBEDDING_MODEL", "from-the-environment")
    assert resolve_embedding_model() == "from-the-environment"


def test_there_is_a_default(monkeypatch):
    monkeypatch.delenv("SIBYL_EMBEDDING_MODEL", raising=False)
    assert resolve_embedding_model() == DEFAULT_EMBEDDING_MODEL


def test_the_same_model_is_built_once(monkeypatch):
    # 90 MB of weights per call is what the first live run actually cost.
    monkeypatch.setenv("SIBYL_EMBEDDING_MODEL", "some-model")
    assert build_embedder() is build_embedder()


def test_the_cache_follows_the_environment(monkeypatch):
    """Changing the model must change the embedder, cache or no cache.

    Caching on the unresolved `None` would key every environment-selected model
    to one entry, and this is the test that would fail — with the old model
    returned silently, producing results that still look like results.
    """
    monkeypatch.setenv("SIBYL_EMBEDDING_MODEL", "model-a")
    first = build_embedder()
    monkeypatch.setenv("SIBYL_EMBEDDING_MODEL", "model-b")
    assert build_embedder().name == "model-b" != first.name


# ── the maths: cosine similarity has known answers ────────────────────────────


def test_identical_vectors_score_one():
    m = np.array([[1.0, 2.0, 3.0]])
    _, scores = cosine_top_k(m, np.array([1.0, 2.0, 3.0]), k=1)
    assert scores[0] == pytest.approx(1.0)


def test_scaled_vectors_still_score_one():
    # Cosine is about direction only; a vector ten times longer is the same point.
    m = np.array([[1.0, 2.0, 3.0]])
    _, scores = cosine_top_k(m, np.array([10.0, 20.0, 30.0]), k=1)
    assert scores[0] == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    m = np.array([[0.0, 1.0]])
    _, scores = cosine_top_k(m, np.array([1.0, 0.0]), k=1)
    assert scores[0] == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one():
    m = np.array([[1.0, 0.0]])
    _, scores = cosine_top_k(m, np.array([-1.0, 0.0]), k=1)
    assert scores[0] == pytest.approx(-1.0)


def test_results_come_back_best_first():
    # Row 1 is the query exactly, row 2 is 45° away, row 0 is orthogonal.
    m = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    indices, scores = cosine_top_k(m, np.array([1.0, 0.0]), k=3)
    assert list(indices) == [1, 2, 0]
    assert list(scores) == sorted(scores, reverse=True)


def test_asking_for_more_than_exists_returns_everything():
    m = np.array([[1.0, 0.0], [0.0, 1.0]])
    indices, _ = cosine_top_k(m, np.array([1.0, 0.0]), k=50)
    assert len(indices) == 2


def test_empty_matrix_returns_nothing():
    indices, scores = cosine_top_k(np.empty((0, 3)), np.array([1.0, 0.0, 0.0]), k=5)
    assert len(indices) == 0 and len(scores) == 0


def test_zero_vector_does_not_produce_nan():
    # A row of zeros has no direction. The floor in `unit` keeps it at score 0
    # rather than letting 0/0 poison the sort with a NaN that compares false to
    # everything and lands wherever argpartition happens to leave it.
    m = np.array([[0.0, 0.0], [1.0, 0.0]])
    _, scores = cosine_top_k(m, np.array([1.0, 0.0]), k=2)
    assert not np.isnan(scores).any()


def test_unit_rows_have_length_one():
    lengths = np.linalg.norm(unit(np.array([[3.0, 4.0], [1.0, 1.0]])), axis=1)
    assert lengths == pytest.approx([1.0, 1.0])


# ── the store: what goes in comes back ────────────────────────────────────────

WEEKLY = "Strong seasonality detected weekly. 3 anomalies flagged."
MONTHLY = "Monthly frequency, monthly seasonality, series is stationary."


def _remember(job_id: str, text: str, model_name: str, embedder) -> str:
    return remember(job_id, text, model_name, {"target_column": "sales"}, embedder)


def test_a_remembered_memory_is_recalled(db, stub_embedder):
    _remember("job-1", WEEKLY, "prophet", stub_embedder)
    hits = recall(WEEKLY, k=5, embedder=stub_embedder)
    assert [h.job_id for h in hits] == ["job-1"]


def test_an_exact_match_scores_one(db, stub_embedder):
    _remember("job-1", WEEKLY, "prophet", stub_embedder)
    assert recall(WEEKLY, k=1, embedder=stub_embedder)[0].score == pytest.approx(1.0)


def test_the_closer_memory_ranks_first(db, stub_embedder):
    _remember("weekly-job", WEEKLY, "prophet", stub_embedder)
    _remember("monthly-job", MONTHLY, "ets", stub_embedder)

    hits = recall("weekly seasonality with anomalies", k=2, embedder=stub_embedder)
    assert hits[0].job_id == "weekly-job"
    assert hits[0].score > hits[1].score


def test_the_model_that_was_run_comes_back_with_the_hit(db, stub_embedder):
    # The whole point of the feature: not "here is a similar series" but "here is
    # what we ran on a similar series".
    _remember("job-1", MONTHLY, "ets", stub_embedder)
    assert recall(MONTHLY, k=1, embedder=stub_embedder)[0].model_name == "ets"


def test_k_caps_the_number_of_hits(db, stub_embedder):
    for i in range(5):
        _remember(f"job-{i}", f"{WEEKLY} {i}", "prophet", stub_embedder)
    assert len(recall(WEEKLY, k=2, embedder=stub_embedder)) == 2


def test_recall_on_an_empty_store_returns_nothing(db, stub_embedder):
    assert recall(WEEKLY, k=5, embedder=stub_embedder) == []


def test_reindexing_a_job_replaces_its_memory(db, stub_embedder):
    _remember("job-1", WEEKLY, "prophet", stub_embedder)
    _remember("job-1", MONTHLY, "ets", stub_embedder)

    hits = recall(WEEKLY, k=10, embedder=stub_embedder)
    assert len(hits) == 1                 # not two rows for one job
    assert hits[0].model_name == "ets"    # and it is the newer one


def test_another_embedding_space_is_not_searched(db, stub_embedder):
    """Vectors from a different model must not be compared against these.

    This is the failure mode worth a test of its own: the numbers would still be
    in [-1, 1] and would still sort, so nothing would look broken — the ranking
    would just be noise.
    """
    from tests.conftest import WordCountEmbedder

    _remember("job-1", WEEKLY, "prophet", stub_embedder)
    assert recall(WEEKLY, k=5, embedder=WordCountEmbedder(name="some-other-model")) == []


def test_the_stored_vector_survives_the_round_trip(db, stub_embedder):
    """float32 bytes in, the same float32 out — exactly, not approximately."""
    from sqlalchemy import select

    from sibyl.db.engine import session_scope
    from sibyl.models.memory import Memory

    _remember("job-1", WEEKLY, "prophet", stub_embedder)
    expected = stub_embedder.encode([WEEKLY])[0]

    with session_scope() as session:
        row = session.scalars(select(Memory)).one()

    assert np.array_equal(row.vector, expected)
    assert row.dim == len(expected)


# ── the document, and the worker's view of it ─────────────────────────────────

FAKE_RESULT = {
    "summary": WEEKLY,
    "diagnosis": {
        "target_column": "sales",
        "profile": {"detected_frequency": "daily", "row_count": 365},
    },
}


def test_the_embedded_document_is_the_diagnosis_digest():
    # And nothing else: the model name is a column, not a token, so a search for
    # a kind of series is not pulled toward the wording of the answer.
    assert job_document(FAKE_RESULT) == WEEKLY


def test_remember_job_carries_the_metadata_a_hit_needs(db, stub_embedder):
    params = {"model_name": "ets", "horizon": 14, "conformal": True}
    remember_job("job-1", params, FAKE_RESULT, stub_embedder)

    hit = recall(WEEKLY, k=1, embedder=stub_embedder)[0]
    assert hit.model_name == "ets"
    assert hit.meta == {
        "target_column": "sales", "horizon": 14, "conformal": True,
        "frequency": "daily", "row_count": 365,
    }
