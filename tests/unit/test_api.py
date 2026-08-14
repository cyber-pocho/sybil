import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="API layer requires the [api] extra")

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
