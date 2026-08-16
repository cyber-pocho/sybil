# Sibyl

AI-powered time-series forecasting and analytics service. Built around a diagnostic pipeline that understands your data before touching a model.

## What it does

Sibyl ingests raw time-series DataFrames and runs a layered diagnostic suite: profiling, anomaly detection, stationarity, and seasonality. This before handing clean, characterised data to a forecasting layer. The goal is to make the pre-modelling step explicit and inspectable rather than hidden inside a training loop.

## Status

| Layer | State |
|---|---|
| Diagnosis (`src/diagnosis/`) | ✅ Implemented, fully covered |
| Forecasting (`src/forecasting/`) | ✅ Prophet, ETS, split-conformal intervals |
| HTTP API (`src/sibyl/api/`) | ✅ `/health`, `/diagnose`, `/forecast` (async jobs) |
| Agent (`src/sibyl/agents/`) | ✅ LangGraph model-selection agent, any LLM provider |
| Tasks (`src/sibyl/tasks/`) | ✅ Celery worker running forecast jobs |
| Persistence (`src/sibyl/db/`, `models/`) | ✅ SQLAlchemy + Alembic, one `jobs` table |
| Vector search | ⛔ Not started — empty package |

## Project layout

```
src/
├── diagnosis/          # Data understanding pipeline
│   ├── profiler.py     # Column typing, frequency detection, missing values, basic stats
│   ├── stationarity.py # ADF + KPSS tests, recommended differencing order
│   ├── seasonality.py  # FFT + ACF seasonality detection, strength score
│   ├── anomalies.py    # Z-score and Isolation Forest anomaly detection
│   ├── pipeline.py     # run_full_diagnosis() — the four stages, wired together
│   └── schemas.py      # Pydantic output models for all of the above
├── forecasting/        # Forecast models behind one interface
│   ├── base.py         # BaseForecaster ABC: fit / predict / name / card
│   ├── prophet_model.py# Meta's additive decomposition model
│   ├── ets_model.py    # State-space Error-Trend-Seasonal (statsmodels)
│   ├── conformal.py    # Split-conformal prediction intervals for any forecaster
│   ├── schemas.py      # ForecastResult, ModelCard
│   └── registry.py     # MODELS by name — shared by the agent and the worker
└── sibyl/              # Application layer
    ├── api/app.py      # FastAPI factory — the service entry point
    ├── agents/
    │   ├── forecaster_agent.py  # LangGraph agent: diagnose → pick a model → explain
    │   └── llm.py               # Provider registry — the only file naming a vendor
    ├── services/
    │   └── forecasting.py       # diagnose + fit + predict; no db, no queue
    ├── tasks/
    │   ├── celery_app.py        # the Celery app
    │   └── forecast.py          # the worker: run a job, record what happened
    ├── models/job.py   # the Job ORM model
    └── db/             # engine, session scope, declarative base

alembic/                # Migrations; 0001 creates the jobs table

tests/
├── unit/               # Per-module tests; the agent runs against a stub LLM
└── integration/        # End-to-end run_full_diagnosis coverage

scripts/
└── run_agent_live.py   # Run the agent against a real model, any provider
```

244 tests, no skips. `ruff check` clean.

## Diagnostic pipeline

`run_full_diagnosis()` is the single entry point. Each stage returns a typed Pydantic model defined in `src/diagnosis/schemas.py`.

```python
from diagnosis.pipeline import run_full_diagnosis

report = run_full_diagnosis(df, target_column="sales")   # target defaults to first numeric column
print(report.to_summary())
```

```
Dataset: 730 rows, daily frequency, Jan 2023 to Dec 2024.
Target column: 'sales' (mean: 156.0, std: 39.1).
5 missing values detected (0.7%).
Series is non-stationary (ADF p=0.72). First differencing recommended.
Strong seasonality detected (period=7, strength=0.89).
37 anomalies flagged (isolation_forest method).
```

**Stage order matters.** Anomalies are detected *second*, not last, because outliers break the measurements that follow: a few spikes dominate the FFT power budget and drag the ACF below its significance band, so a clean weekly cycle reports as no seasonality at all. Seasonality therefore runs on a de-spiked copy of the series, with flagged points interpolated to preserve phase. Stationarity deliberately runs on the raw series — a unit root is a property of the trend, not of a few spikes.

### 1. Profiler

```python
from diagnosis.profiler import profile

report = profile(df)
# report.datetime_column    — which column is the time axis
# report.detected_frequency — "daily" | "weekly" | "monthly" | "hourly" | "irregular"
# report.numeric_columns    — candidate forecast targets
# report.duplicate_timestamps
# report.columns[i].missing_pct
# report.columns[i].stats   — mean, std, min, max, skew, kurtosis
```

Detects the datetime column via parse-attempt heuristic (qualifies when ≥80 % of rows parse cleanly). Frequency comes from the median inter-row delta matched against named bands, gated on the median absolute deviation of those deltas — gaps of 2 d, 7 d and 36 d have a median of exactly 7 d and are reported as `irregular` rather than as clean weekly data.

### 2. Anomaly detection

```python
from diagnosis.anomalies import detect_anomalies

result = detect_anomalies(series, method="auto")
# result.method_used      — "zscore" or "isolation_forest"
# result.anomaly_count
# result.anomaly_indices
# result.details[i].score — z-score or IF decision-function value
```

`method="auto"` runs Shapiro-Wilk on up to 500 samples to test for normality, then picks z-score (Gaussian data) or Isolation Forest (everything else). Both methods are available directly as `zscore_anomalies()` and `isolation_forest_anomalies()`.

### 3. Stationarity

```python
from diagnosis.stationarity import check_stationarity

result = check_stationarity(series)
# result.is_stationary            — True when ADF p<0.05 AND KPSS p>0.05
# result.adf_pvalue
# result.kpss_pvalue
# result.recommended_differencing — 0, 1, or 2
```

Runs ADF (H0: unit root) and KPSS (H0: stationary) jointly because each can be fooled alone. Automatically tests d=1 and d=2 to recommend the minimum differencing order.

### 4. Seasonality

```python
from diagnosis.seasonality import detect_seasonality

result = detect_seasonality(series, freq="D")
# result.has_seasonality
# result.detected_periods     — e.g. [7, 14] for weekly + biweekly
# result.dominant_period
# result.seasonality_strength — 0–1, fraction of detrended variance in seasonal frequencies
```

Linearly detrends the series, computes the FFT power spectrum, then confirms candidate periods via ACF (Bartlett ±2/√n band). A peak must appear in both FFT and ACF to be accepted — this cuts spectral leakage false positives.

## Forecasting

Every model implements `BaseForecaster`: `fit(series)`, `predict(horizon)`, `name`, and `card`. All return the same `ForecastResult` with 80 % and 95 % intervals.

```python
from forecasting.prophet_model import ProphetForecaster
from forecasting.conformal import ConformalWrapper

model = ProphetForecaster()
model.fit(series)                  # series must carry a DatetimeIndex
result = model.predict(horizon=30)
# result.point_forecast, result.lower_80/upper_80, result.lower_95/upper_95
```

| Model | Intervals | Handles NaNs | Min samples |
|---|---|---|---|
| `ProphetForecaster` | native (posterior quantiles) | yes | 100 |
| `ETSForecaster` | native (analytical state-space) | no | 24 |
| `ConformalWrapper(model)` | split-conformal | inherits | 2× base, min 50 |

`ConformalWrapper` replaces a model's native intervals with distribution-free ones: it trains the base model on the first 70 %, scores absolute residuals on the last 30 %, and applies the exact finite-sample quantile `⌈(1-α)(n+1)⌉` (Angelopoulos & Bates 2021, Thm 1). When the calibration set is too small to bound the level, it returns infinite width rather than silently under-covering. Empirical coverage at 80 % and 95 % is asserted in the test suite.

## Agent

`src/sibyl/agents/` is a LangGraph agent that runs the diagnosis, reads the
`ModelCard` of each forecaster, picks one, and explains the choice in terms of what
the diagnosis found. Model selection is the LLM's decision, not a rule table — the
cards are the evidence it reasons over.

```python
from sibyl.agents.forecaster_agent import run_agent

run = run_agent(series, horizon=30)   # series needs a DatetimeIndex
run.explanation                       # why it picked that model, in its own words
run.model_used                        # e.g. "prophet" or "conformal(ets)"
run.forecast                          # ForecastResult
run.diagnosis                         # FullDiagnosisReport
run.messages                          # full transcript, for debugging
```

The graph is the standard two-node ReAct loop — `START → agent ⇄ tools → END` —
with three tools: `diagnose`, `list_models`, and `run_forecast`. It runs until the
model stops calling tools, bounded by `recursion_limit`.

`run_forecast` refuses a model whose `ModelCard.min_samples_required` exceeds the
usable history, naming the models that do fit. The floor was advertised on every card
and enforced by nothing — `ProphetForecaster.fit()` will fit 60 points against its own
stated minimum of 100 — so a live model read the number, said it out loud, and fitted
anyway. Refusals return as text rather than raising, so the agent picks again instead
of aborting the graph.

Run it yourself against any provider:

```bash
SIBYL_LLM_PROVIDER=ollama SIBYL_LLM_MODEL=qwen2.5:7b python scripts/run_agent_live.py
```

Two fixtures that discriminate on different card fields — one on `handles_missing`,
one on `min_samples_required` — plus the full transcript and token counts.

### Choosing a model

The agent is provider-agnostic. It asks two things of a model — that it speaks the
LangChain chat interface, and that it can call tools — and nothing else. Closed or
open weights, hosted or on your own machine, it is the same code path.

Pick a provider with two environment variables:

```bash
pip install -e ".[agents,anthropic]"        # or openai / google / mistral / groq / ollama
export SIBYL_LLM_PROVIDER=anthropic
export SIBYL_LLM_MODEL=claude-opus-5
export ANTHROPIC_API_KEY=sk-ant-...
```

| `SIBYL_LLM_PROVIDER` | Extra | Credential | Example model |
|---|---|---|---|
| `anthropic` | `[anthropic]` | `ANTHROPIC_API_KEY` | `claude-opus-5` |
| `openai` | `[openai]` | `OPENAI_API_KEY` | `gpt-5` |
| `google` | `[google]` | `GOOGLE_API_KEY` | `gemini-2.5-pro` |
| `mistral` | `[mistral]` | `MISTRAL_API_KEY` | `mistral-large-latest` |
| `groq` | `[groq]` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `ollama` | `[ollama]` | *none* | `llama3.1:8b` |
| `openai_compatible` | `[openai]` | *usually none* | whatever your server serves |

`ollama` runs open weights locally. `openai_compatible` covers anything exposing an
OpenAI-shaped `/v1` endpoint — vLLM, llama.cpp's server, LM Studio, TGI, OpenRouter,
Together — via `SIBYL_LLM_BASE_URL`:

```bash
export SIBYL_LLM_PROVIDER=openai_compatible
export SIBYL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SIBYL_LLM_BASE_URL=http://localhost:8000/v1
```

**There is no default provider.** An unset `SIBYL_LLM_PROVIDER` raises rather than
quietly sending a request to somebody's paid endpoint. Vendor-specific settings go
through `**kwargs`, which is also how a model gets anything the registry is too
small to know about:

```python
from sibyl.agents.llm import build_llm

llm = build_llm("anthropic", "claude-opus-5", thinking={"type": "adaptive"})
llm = build_llm("ollama", "llama3.1:8b", num_ctx=8192)
run = run_agent(series, horizon=30, llm=llm)
```

Adding a provider is one row in `_PROVIDERS` (`src/sibyl/agents/llm.py`) plus one
line in `pyproject.toml`. Nothing in `forecaster_agent.py` changes.

### Picking a model that can actually do this

**Tool calling is required, not optional** — the loop *is* the model choosing tools.
The agent needs three calls minimum, and four when it has to recover from a tool
error, so a model that tool-calls *sometimes* is not enough.

Measured on `qwen2.5:7b` over eight local runs:

| Path | Calls | Success |
|---|---|---|
| diagnose → list_models → run_forecast | 3 | ~2 in 3 |
| the above, plus a retry after a refusal | 4 | **0 in 6** |

It reasons correctly on the failing path and then *describes* the tool call in prose
instead of making one. Treat ~7B as good enough to smoke-test the wiring and not for
production; use a frontier hosted model, or a considerably larger local one, for real
work.

Declared capability is not evidence. `qwen2.5-coder:7b` advertises `tools` in Ollama
and returns `tool_calls=[]` with the JSON as text. Probe before trusting it:

```python
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"

llm = ChatOllama(model="<your model>").bind_tools([StructuredTool.from_function(get_weather)])
print(llm.invoke("Weather in Paris? Use the tool.").tool_calls)   # must be non-empty
```

⚠️ The agent has run end to end against a **local Ollama model only**. The hosted
providers are verified as far as constructing a client and binding tools — no live
round-trip. Run `scripts/run_agent_live.py` against yours before trusting it.

## HTTP API

```bash
make run-api        # uvicorn sibyl.api.app:create_app --factory, on :8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `POST /diagnose` | Run the full diagnosis over row-oriented JSON records |
| `POST /forecast` | Queue a forecast job — returns `202` and a job id |
| `GET /forecast/{id}` | Poll that job for status, result, or error |

```bash
curl -X POST localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"data": [{"date": "2024-01-01", "sales": 100}, ...], "target_column": "sales"}'
```

Returns `{"report": <FullDiagnosisReport>, "summary": "<6-line digest>"}`. Input the pipeline cannot diagnose — no numeric columns, or an unknown `target_column` — comes back as 422. Interactive docs at `/docs`.

### Forecast jobs

`/diagnose` runs inline because a diagnosis is milliseconds. Fitting Prophet is
seconds, so `/forecast` does not: it writes a job, hands back an id, and returns
`202`. The work happens in a Celery worker, and the result lands in Postgres where
a client can collect it whenever it likes.

```bash
make migrate        # alembic upgrade head — run before either process starts
make run-api        # the API
make run-worker     # celery -A sibyl.tasks.celery_app worker
```

```bash
curl -X POST localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"data": [...], "target_column": "sales", "model_name": "ets", "horizon": 14}'
# 202  {"id": "9f1c...", "status": "pending", "result": null, "error": null}

curl localhost:8000/forecast/9f1c...
# 200  {"id": "9f1c...", "status": "done", "result": {"diagnosis": ..., "summary": ..., "forecast": ...}}
```

A job is `pending` → `running` → `done` or `failed`. **A failed forecast is a
recorded outcome, not a lost job**: the worker catches whatever the forecaster
raised, writes it to `error`, and the row still reaches a terminal state — a
client polling it never hangs. An unknown `model_name` is the one thing rejected
inline with a `422`, because the caller is still on the phone and can fix it.

```json
{"id": "...", "status": "failed",
 "error": "ValueError: prophet needs at least 100 usable observations and this series has 60. Models that fit: ets."}
```

The queue carries only the job id. Everything the work needs is already in the
row, which keeps messages small and means a retry reads current state rather than
a snapshot from when the message was written.

## Stack

Core install is the numeric stack only. Everything else is an extra, so profiling a DataFrame does not pull in torch.

| Layer | Libraries | Extra | Status |
|---|---|---|---|
| Data / numerics | pandas, NumPy, SciPy, scikit-learn | *(core)* | in use |
| Forecasting | Prophet, statsmodels | *(core)* | in use |
| Schemas | Pydantic v2 | *(core)* | in use |
| API | FastAPI, uvicorn | `[api]` | in use |
| Agent | LangGraph, langchain-core | `[agents]` | in use |
| LLM provider | one of langchain-{anthropic,openai,google-genai,mistralai,groq,ollama} | `[anthropic]` … `[ollama]` | in use, pick one |
| Vector search | FAISS, sentence-transformers, PyTorch | `[vectorsearch]` | planned |
| Task queue | Celery + Redis | `[workers]` | in use |
| Database | SQLAlchemy, psycopg, Alembic | `[db]` | in use |
| Visualisation | Plotly, matplotlib | `[viz]` | planned |
| Observability / payments | Weights & Biases, Stripe | `[ops]` | planned |

Dependencies carry upper version bounds. This is deliberate: an unpinned `mapie` is how a removed class (`MapieRegressor`, dropped in MAPIE 1.x) quietly turned a wrapper into dead code that still imported cleanly.

## Getting started

```bash
# Install (core + dev tools + API + agent + persistence + worker, no model provider)
make install          # pip install -e ".[dev,api,agents,db,workers]"

# Add whichever provider you intend to use
pip install -e ".[anthropic]"   # or openai / google / mistral / groq / ollama

# Run tests
make test             # pytest → tests/unit + tests/integration

# Lint
make lint             # ruff check src tests

# Start API (dev)
make run-api          # uvicorn with --reload on :8000

# Create the schema (SQLite by default; set DATABASE_URL for Postgres)
make migrate          # alembic upgrade head

# Start the worker that runs forecast jobs
make run-worker       # celery -A sibyl.tasks.celery_app worker

# Start the stack
make docker-up        # api + worker + redis + postgres
```

## Environment variables

`.env.example` lists them all.

Read by the agent today:

| Variable | Purpose |
|---|---|
| `SIBYL_LLM_PROVIDER` | which provider to build a model from — no default |
| `SIBYL_LLM_MODEL` | the vendor's own model id |
| `SIBYL_LLM_MAX_TOKENS` | optional output cap, mapped to each vendor's own name |
| `SIBYL_LLM_BASE_URL` | endpoint for `openai_compatible`; optional for `ollama` |
| `<PROVIDER>_API_KEY` | the credential for the provider you chose, if it needs one |

Read by the persistence and task layers:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL; defaults to `sqlite:///./sibyl.db` so a fresh clone runs with no infrastructure |
| `REDIS_URL` | Celery broker and result backend; defaults to `redis://localhost:6379/0` |

Not read by any code yet: `WANDB_API_KEY`, `STRIPE_SECRET_KEY`,
`FAISS_INDEX_PATH`. `.env` stays optional — the API and `docker compose up` both
start without one.

## CI

GitHub Actions runs on every push to `main` and all PRs: `ruff check src/ tests/`, then `pytest tests/ -v` across both the unit and integration suites. CI installs `[dev,api,agents]` and **no provider package at all** — the agent tests run against a stub LLM, so no vendor credential is needed, and a green CI run is itself the evidence that the agent core carries no vendor dependency. `make lint` runs the identical lint command, so a green local run means a green CI run.
