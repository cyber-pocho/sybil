# Sibyl

AI-powered time-series forecasting and analytics service. Built around a diagnostic pipeline that understands your data before touching a model.

## What it does

Sibyl ingests raw time-series DataFrames and runs a layered diagnostic suite — profiling, stationarity, seasonality, and anomaly detection — before handing clean, characterised data to a forecasting layer. The goal is to make the pre-modelling step explicit and inspectable rather than hidden inside a training loop.

## Project layout

```
src/
├── diagnosis/          # Data understanding pipeline (implemented)
│   ├── profiler.py     # Column typing, frequency detection, missing values, basic stats
│   ├── stationarity.py # ADF + KPSS tests, recommended differencing order
│   ├── seasonality.py  # FFT + ACF seasonality detection, strength score
│   ├── anomalies.py    # Z-score and Isolation Forest anomaly detection
│   └── schemas.py      # Pydantic output models for all of the above
└── sibyl/              # Application layer (in progress)
    ├── api/            # FastAPI app
    ├── agents/         # LangGraph agents
    ├── forecasting/    # Prophet, statsmodels, MAPIE models
    ├── services/       # Business logic
    ├── tasks/          # Celery workers
    └── db/             # SQLAlchemy models, Alembic migrations

tests/
└── unit/
    ├── test_profiler.py
    ├── test_stationarity_seasonality.py
    └── test_anomalies.py
```

## Diagnostic pipeline

Each function returns a typed Pydantic model defined in `src/diagnosis/schemas.py`.

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

Detects the datetime column via parse-attempt heuristic (qualifies when ≥80 % of rows parse cleanly). Frequency is inferred from the median inter-row delta matched against named bands.

### 2. Stationarity

```python
from diagnosis.stationarity import test_stationarity

result = test_stationarity(series)
# result.is_stationary            — True when ADF p<0.05 AND KPSS p>0.05
# result.adf_pvalue
# result.kpss_pvalue
# result.recommended_differencing — 0, 1, or 2
```

Runs ADF (H0: unit root) and KPSS (H0: stationary) jointly because each can be fooled alone. Automatically tests d=1 and d=2 to recommend the minimum differencing order.

### 3. Seasonality

```python
from diagnosis.seasonality import detect_seasonality

result = detect_seasonality(series, freq="D")
# result.has_seasonality
# result.detected_periods     — e.g. [7, 14] for weekly + biweekly
# result.dominant_period
# result.seasonality_strength — 0–1, fraction of detrended variance in seasonal frequencies
```

Linearly detrends the series, computes the FFT power spectrum, then confirms candidate periods via ACF (Bartlett ±2/√n band). A peak must appear in both FFT and ACF to be accepted — this cuts spectral leakage false positives.

### 4. Anomaly detection

```python
from diagnosis.anomalies import detect_anomalies

result = detect_anomalies(series, method="auto")
# result.method_used      — "zscore" or "isolation_forest"
# result.anomaly_count
# result.anomaly_indices
# result.details[i].score — z-score or IF decision-function value
```

`method="auto"` runs Shapiro-Wilk on up to 500 samples to test for normality, then picks z-score (Gaussian data) or Isolation Forest (everything else). Both methods are available directly as `zscore_anomalies()` and `isolation_forest_anomalies()`.

## Stack

| Layer | Libraries |
|---|---|
| API | FastAPI, uvicorn, Pydantic v2 |
| Agents | LangGraph, LangChain, langchain-anthropic |
| Vector search | FAISS, sentence-transformers |
| Forecasting | Prophet, statsmodels, MAPIE, scikit-learn |
| ML / numerics | PyTorch, NumPy, SciPy, pandas |
| Task queue | Celery + Redis |
| Database | SQLAlchemy, asyncpg, Alembic (PostgreSQL) |
| Observability | Weights & Biases |
| Payments | Stripe |

## Getting started

```bash
# Install
make install          # pip install -e ".[dev]"

# Copy and fill in secrets
cp .env.example .env

# Run tests
make test             # pytest tests/unit/ -v

# Lint
make lint             # ruff check src/ tests/

# Start API (dev)
make run-api          # uvicorn with --reload on :8000

# Start full stack
make docker-up        # api + worker + redis + postgres
```

## Environment variables

See `.env.example` for the full list. Required before running:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API access for agents |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Celery broker and result backend |
| `WANDB_API_KEY` | Experiment tracking |
| `STRIPE_SECRET_KEY` | Billing |
| `FAISS_INDEX_PATH` | Where the vector index is persisted on disk |

## CI

GitHub Actions runs on every push to `main` and all PRs: lint (`ruff`) then `pytest tests/unit/`.
