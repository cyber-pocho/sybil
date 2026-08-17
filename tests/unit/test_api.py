import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="API layer requires the [api] extra")
# app.py imports the persistence and queue layers for /forecast, so importing it
# needs those extras too — without these the module fails to import and the whole
# file errors out at collection instead of skipping.
pytest.importorskip("sqlalchemy", reason="API layer requires the [db] extra")
pytest.importorskip("celery", reason="API layer requires the [workers] extra")

from fastapi.testclient import TestClient  # noqa: E402

from sibyl.api.app import create_app  # noqa: E402

RNG = np.random.default_rng(3)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(scope="module")
def payload() -> list[dict]:
    """A year of daily data with a weekly cycle — enough for every stage to run."""
    n = 365
    t = np.arange(n)
    sales = 100.0 + 0.2 * t + 8.0 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 2.0, n)
    df = pd.DataFrame({
        "date":  pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "sales": sales,
    })
    return df.to_dict("records")


# ── health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── diagnose ──────────────────────────────────────────────────────────────────

def test_diagnose_returns_200(client, payload):
    assert client.post("/diagnose", json={"data": payload}).status_code == 200


def test_diagnose_picks_first_numeric_column(client, payload):
    body = client.post("/diagnose", json={"data": payload}).json()
    assert body["report"]["target_column"] == "sales"


def test_diagnose_detects_daily_frequency(client, payload):
    body = client.post("/diagnose", json={"data": payload}).json()
    assert body["report"]["profile"]["detected_frequency"] == "daily"


def test_diagnose_finds_weekly_seasonality(client, payload):
    body = client.post("/diagnose", json={"data": payload}).json()
    assert body["report"]["seasonality"]["dominant_period"] == 7


def test_diagnose_returns_six_line_summary(client, payload):
    body = client.post("/diagnose", json={"data": payload}).json()
    assert len(body["summary"].splitlines()) == 6


def test_diagnose_honours_explicit_target(client, payload):
    rows = [{**row, "units": 5.0} for row in payload]
    body = client.post("/diagnose", json={"data": rows, "target_column": "units"}).json()
    assert body["report"]["target_column"] == "units"


# ── input the caller got wrong → 422, not 500 ─────────────────────────────────

def test_unknown_target_column_is_422(client, payload):
    r = client.post("/diagnose", json={"data": payload, "target_column": "nope"})
    assert r.status_code == 422


def test_no_numeric_columns_is_422(client):
    r = client.post("/diagnose", json={"data": [{"label": "a"}, {"label": "b"}]})
    assert r.status_code == 422


def test_empty_data_is_rejected(client):
    # min_length=1 on the field — pydantic rejects this before the pipeline runs
    assert client.post("/diagnose", json={"data": []}).status_code == 422


# ── forecast: accepted as a job, not run inline ───────────────────────────────

def _post_forecast(client, payload, **over):
    body = {"data": payload, "target_column": "sales", "model_name": "ets", "horizon": 7} | over
    return client.post("/forecast", json=body)


def test_forecast_is_accepted_with_202(db, no_broker, payload):
    # A fresh client per test: the module-scoped one would outlive the db fixture.
    client = TestClient(create_app())
    r = _post_forecast(client, payload)
    assert r.status_code == 202
    assert r.json()["status"] == "pending"


def test_forecast_returns_a_job_id(db, no_broker, payload):
    client = TestClient(create_app())
    job_id = _post_forecast(client, payload).json()["id"]
    assert job_id


def test_forecast_dispatches_the_job_it_wrote(db, no_broker, payload):
    # The id handed to the caller and the id put on the queue must be the same,
    # or the worker runs one job while the client polls another.
    client = TestClient(create_app())
    job_id = _post_forecast(client, payload).json()["id"]
    assert no_broker == [job_id]


def test_unknown_model_is_422_before_any_job_is_written(db, no_broker, payload):
    client = TestClient(create_app())
    r = _post_forecast(client, payload, model_name="arima")
    assert r.status_code == 422
    assert "Unknown model" in r.json()["detail"]
    assert no_broker == []   # nothing queued


def test_forecast_empty_data_is_rejected(db, no_broker):
    # Named for its endpoint: the /diagnose case is already
    # `test_empty_data_is_rejected`, and reusing the name would shadow it — the
    # same way a duplicate definition silently disabled a stationarity test here
    # before anyone was running the suite.
    client = TestClient(create_app())
    assert client.post("/forecast", json={"data": []}).status_code == 422


def test_non_positive_horizon_is_rejected(db, no_broker, payload):
    client = TestClient(create_app())
    assert _post_forecast(client, payload, horizon=0).status_code == 422


# ── polling a job ─────────────────────────────────────────────────────────────

def test_pending_job_reports_pending(db, no_broker, payload):
    client = TestClient(create_app())
    job_id = _post_forecast(client, payload).json()["id"]
    r = client.get(f"/forecast/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert r.json()["result"] is None


def test_unknown_job_is_404(db, no_broker):
    client = TestClient(create_app())
    assert client.get("/forecast/no-such-job").status_code == 404


def test_full_round_trip_post_then_work_then_poll(db, no_broker, payload):
    """The whole contract: accept, run the worker, read the result back.

    This is the one test that proves the pieces line up — the params the endpoint
    stored are the params the service accepts, and the result it wrote survives a
    round trip through the JSON column.
    """
    from sibyl.tasks.forecast import run_forecast_job

    client = TestClient(create_app())
    job_id = _post_forecast(client, payload).json()["id"]

    run_forecast_job(job_id)   # what the Celery worker would do

    body = client.get(f"/forecast/{job_id}").json()
    assert body["status"] == "done"
    assert len(body["result"]["forecast"]["point_forecast"]) == 7
    assert body["result"]["diagnosis"]["target_column"] == "sales"
    assert body["error"] is None


# ── search over past forecasts ────────────────────────────────────────────────

@pytest.fixture
def search_client(db, no_broker, stub_embedder, monkeypatch) -> TestClient:
    """A client whose /search uses the stub embedder instead of downloading weights."""
    from sibyl.api import app as app_module

    monkeypatch.setattr(app_module, "build_embedder", lambda *a, **k: stub_embedder)
    return TestClient(create_app())


def _remember_one(job_id: str, text: str, model_name: str, embedder) -> None:
    from sibyl.vectorsearch.store import remember

    remember(job_id, text, model_name, {}, embedder)


def test_search_by_text_finds_a_past_forecast(search_client, stub_embedder):
    _remember_one("old-job", "Weekly seasonality, 3 anomalies flagged.", "prophet", stub_embedder)

    r = search_client.post("/search", json={"query": "weekly seasonality with anomalies"})
    assert r.status_code == 200
    assert r.json()["hits"][0]["job_id"] == "old-job"
    assert r.json()["hits"][0]["model_name"] == "prophet"


def test_search_by_data_diagnoses_the_series_first(search_client, stub_embedder, payload):
    _remember_one("old-job", "Weekly seasonality detected.", "prophet", stub_embedder)

    r = search_client.post("/search", json={"data": payload, "target_column": "sales"})
    assert r.status_code == 200
    # The echoed query is the digest the endpoint built, not anything the caller
    # wrote — which is what makes a surprising ranking debuggable.
    assert len(r.json()["query"].splitlines()) == 6


def test_search_respects_k(search_client, stub_embedder):
    for i in range(4):
        _remember_one(f"job-{i}", f"Weekly seasonality {i}.", "prophet", stub_embedder)
    assert len(search_client.post("/search", json={"query": "weekly", "k": 2}).json()["hits"]) == 2


def test_search_with_nothing_to_search_by_is_422(search_client):
    assert search_client.post("/search", json={"k": 3}).status_code == 422


def test_search_with_both_query_and_data_is_422(search_client, payload):
    # A caller who sent both has a bug; picking one silently hides it.
    r = search_client.post("/search", json={"query": "weekly", "data": payload})
    assert r.status_code == 422


def test_search_on_an_empty_store_returns_no_hits(search_client):
    r = search_client.post("/search", json={"query": "weekly"})
    assert r.status_code == 200
    assert r.json()["hits"] == []


def test_search_without_the_extra_installed_is_503(db, no_broker, monkeypatch):
    """Not a 500: the service is fine, this one capability just is not installed."""
    from sibyl.api import app as app_module

    def _explode(*args, **kwargs):
        raise ImportError('Install it with: pip install -e ".[vectorsearch]"')

    monkeypatch.setattr(app_module, "build_embedder", _explode)
    r = TestClient(create_app()).post("/search", json={"query": "weekly"})
    assert r.status_code == 503
    assert "vectorsearch" in r.json()["detail"]


def test_round_trip_records_a_failure(db, no_broker, payload):
    from sibyl.tasks.forecast import run_forecast_job

    client = TestClient(create_app())
    # 60 points, against prophet's floor of 100: the service refuses, and the
    # failure has to reach the caller through the job row rather than a 500.
    short = payload[:60]
    job_id = _post_forecast(client, short, model_name="prophet").json()["id"]

    run_forecast_job(job_id)

    body = client.get(f"/forecast/{job_id}").json()
    assert body["status"] == "failed"
    assert "needs at least 100" in body["error"]
    assert body["result"] is None
