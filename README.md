# Sibyl

AI-powered time-series forecasting and analytics service. Built around a diagnostic pipeline that understands your data before touching a model.

## What it does

Sibyl ingests raw time-series DataFrames and runs a layered diagnostic suite: profiling, anomaly detection, stationarity, and seasonality. This before handing clean, characterised data to a forecasting layer. The goal is to make the pre-modelling step explicit and inspectable rather than hidden inside a training loop.

## Status

| Layer | State |
|---|---|
| Diagnosis (`src/diagnosis/`) | ✅ Implemented, fully covered |
| Forecasting (`src/forecasting/`) | ✅ Prophet, ETS, split-conformal intervals |
| HTTP API (`src/sibyl/api/`) | 🟡 `/health` and `/diagnose` only |
| Agents, tasks, persistence | ⛔ Not started — empty packages |

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
│   └── schemas.py      # ForecastResult, ModelCard
└── sibyl/              # Application layer
    ├── api/app.py      # FastAPI factory — the service entry point
    ├── agents/         # LangGraph agents      (empty)
    ├── services/       # Business logic        (empty)
    ├── tasks/          # Celery workers        (empty)
    ├── models/         #                       (empty)
    └── db/             # SQLAlchemy, Alembic   (empty)

tests/
├── unit/               # Per-module tests
└── integration/        # End-to-end run_full_diagnosis coverage
```

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

## HTTP API

```bash
make run-api        # uvicorn sibyl.api.app:create_app --factory, on :8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `POST /diagnose` | Run the full diagnosis over row-oriented JSON records |

```bash
curl -X POST localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"data": [{"date": "2024-01-01", "sales": 100}, ...], "target_column": "sales"}'
```

Returns `{"report": <FullDiagnosisReport>, "summary": "<6-line digest>"}`. Input the pipeline cannot diagnose — no numeric columns, or an unknown `target_column` — comes back as 422. Interactive docs at `/docs`.

## Stack

Core install is the numeric stack only. Everything else is an extra, so profiling a DataFrame does not pull in torch.

| Layer | Libraries | Extra | Status |
|---|---|---|---|
| Data / numerics | pandas, NumPy, SciPy, scikit-learn | *(core)* | in use |
| Forecasting | Prophet, statsmodels | *(core)* | in use |
| Schemas | Pydantic v2 | *(core)* | in use |
| API | FastAPI, uvicorn | `[api]` | in use |
| Agents | LangGraph, LangChain, FAISS, PyTorch | `[agents]` | planned |
| Task queue | Celery + Redis | `[workers]` | planned |
| Database | SQLAlchemy, asyncpg, Alembic | `[db]` | planned |
| Visualisation | Plotly, matplotlib | `[viz]` | planned |
| Observability / payments | Weights & Biases, Stripe | `[ops]` | planned |

Dependencies carry upper version bounds. This is deliberate: an unpinned `mapie` is how a removed class (`MapieRegressor`, dropped in MAPIE 1.x) quietly turned a wrapper into dead code that still imported cleanly.

## Getting started

```bash
# Install (core + dev tools + API)
make install          # pip install -e ".[dev,api]"

# Run tests
make test             # pytest → tests/unit + tests/integration

# Lint
make lint             # ruff check src tests

# Start API (dev)
make run-api          # uvicorn with --reload on :8000

# Start the stack
make docker-up        # api + redis + postgres
```

## Environment variables

`.env.example` lists the variables the planned layers will need (`ANTHROPIC_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `WANDB_API_KEY`, `STRIPE_SECRET_KEY`, `FAISS_INDEX_PATH`).

**None of them are read by any code today**, and `.env` is optional — the API and `docker compose up` both start without one. They become required as the agent, database and billing layers land.

## CI

GitHub Actions runs on every push to `main` and all PRs: `ruff check src/ tests/`, then `pytest tests/ -v` across both the unit and integration suites. `make lint` runs the identical lint command, so a green local run means a green CI run.
