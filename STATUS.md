# Sibyl — Project Status Report

**Date:** 2026-08-14
**Branch:** `main` @ `7a05471` (working tree clean)
**Basis:** full read-through, plus a clean-room install, lint and test run

---

## Verdict in one paragraph

The **diagnosis layer is complete and working**. The **forecasting layer is working** —
Prophet, ETS and split-conformal intervals all pass their tests. The **HTTP API exists in
minimal form**: `/health` and `/diagnose`, verified serving under uvicorn. Agents, background
tasks and persistence remain **unstarted**. The repository's defining problem — a broken build
backend that made `pip install -e .` fail, so **CI failed on all 10 commits and never executed
a single test** — has been fixed, along with everything the resulting blind spot had allowed to
accumulate. The suite went from *119 passed / 8 failed / 20 errors / 2 skipped* to **157 passed,
0 failed, 0 skipped**, and `ruff check` from 30 errors to clean.

---

## Current state, verified

Everything below was executed, not inferred. A fresh venv was created and the exact CI steps run
against it:

```
pip install -e ".[dev,api]"     → exit 0
ruff check src/ tests/          → All checks passed!
pytest tests/                   → 157 passed, 4 warnings
```

The API was additionally booted under the real entry point (`uvicorn sibyl.api.app:create_app
--factory`) and both endpoints returned 200 with correct payloads. `docker compose config`
validates, including from a clone with no `.env` present.

| Metric | Before (`0fb8648`) | Now (`7a05471`) |
|---|---|---|
| `pip install -e .` | ❌ `BackendUnavailable` | ✅ exit 0 |
| CI outcome | ❌ 10/10 runs failed at install | ✅ all three steps pass locally |
| Tests executed in CI | **0, ever** | 157 |
| Test results | 119 pass / 8 fail / 20 error / 2 skip | **157 pass / 0 fail / 0 skip** |
| `ruff check` | 30 errors | **0** |
| Source LOC | 1,383 | 1,368 |
| Test LOC | 1,226 | 1,290 |

Source is marginally smaller: deleting `MAPIEWrapper` and `_PrefitStub` roughly cancelled out the
new API module. Tests grew by 64 lines net — the API added 10, MAPIE's 3 went away (2 of which
had always silently skipped).

---

## Component inventory

| Component | LOC | Tests | Status |
|---|---|---|---|
| `diagnosis/profiler.py` | 169 | 20 | ✅ Dispersion gate added — irregular spacing no longer reads as weekly. |
| `diagnosis/stationarity.py` | 96 | 16 | ✅ Renamed `test_stationarity` → `check_stationarity`. |
| `diagnosis/seasonality.py` | 106 | ↑ | ✅ Unchanged; now fed a de-spiked series by the pipeline. |
| `diagnosis/anomalies.py` | 115 | 16 | ✅ Unchanged. |
| `diagnosis/pipeline.py` | 91 | 23 | ✅ Reordered: profile → anomalies → stationarity → seasonality. |
| `diagnosis/schemas.py` | 159 | — | ✅ `StrEnum`, long lines wrapped. |
| `forecasting/base.py` | 65 | — | ✅ Unchanged. |
| `forecasting/prophet_model.py` | 143 | 22 | ✅ Unchanged behaviour. |
| `forecasting/ets_model.py` | 206 | 28 | ✅ **Fixed** — was completely broken. |
| `forecasting/conformal.py` | 133 | 22 | ✅ `ConformalWrapper` only; MAPIE removed. |
| `sibyl/api/app.py` | 60 | 10 | 🟡 `/health` + `/diagnose`. Verified under uvicorn. |
| `sibyl/{agents,services,tasks,db,models}` | 0 | — | ⛔ Not started. |

---

## What was fixed

### 1. Build backend — the blocker behind everything else

`pyproject.toml` declared `build-backend = "setuptools.backends.legacy:build"`, a module that
does not exist. Now `setuptools.build_meta`.

This one line is why CI had failed on **every commit since the first**, always at "Install
dependencies", with Lint and Test skipped. It is also why the other bugs survived: the tests
that would have caught them were written, committed, and never run.

### 2. ETS forecaster — was raising `KeyError` on every call

`ets_model.py` read `frame["mean_ci_lower"]`/`["mean_ci_upper"]`. That is the **SARIMAX**
column naming; `ETSResults.get_prediction().summary_frame()` returns `mean`, `pi_lower`,
`pi_upper`, and has since statsmodels 0.12 — so this was never correct. Fixed to `pi_*`,
clearing 19 errors and 1 failure. The duplicated `get_prediction()` call was also collapsed
to one, since alpha is applied later in `summary_frame()`.

### 3. MAPIE — deleted

`MAPIEWrapper` imported `mapie.regression.MapieRegressor`, removed in MAPIE 1.x. The
`try/except ImportError` swallowed the failure, so `_MAPIE_AVAILABLE` silently became `False`,
its two tests silently skipped, and the class raised *"pip install mapie"* even with MAPIE
installed. Removed the class, `_PrefitStub`, its three tests and the dependency.

`ConformalWrapper` already implements the same split-conformal maths, its empirical coverage is
asserted at both 80 % and 95 %, and it needs no sklearn adapter. MAPIE earns a place back only
if Jackknife+ or cross-conformal variants are actually wanted — `TimeSeriesRegressor` is the
modern entry point.

### 4. Outliers vs. seasonality — the one architectural fix

Three spikes in a 730-point series reduced seasonality detection to
`has_seasonality=False, strength=0.00` on a textbook weekly cycle. Spike variance dominates the
FFT power budget and drags the ACF at lag 7 below the Bartlett ±2/√n band. NaNs were never the
problem — outliers were.

The pipeline detected those spikes at **step 4**, one step after they had already ruined the
step-3 measurement. Anomalies now run **second**, and seasonality runs on a copy with flagged
points interpolated — interpolated rather than dropped, because dropping shifts every later
observation and smears the very periodicity being measured.

Measured end-to-end through `run_full_diagnosis` on a 730-point daily series:

| Series | Before | After |
|---|---|---|
| clean | period 7, strength 0.88 | period 7, strength 0.88 |
| + 3 spikes | **none detected, 0.00** | **period 7, strength 0.89** |

The clean case is unchanged, which is the point: de-spiking costs nothing when there is nothing
to de-spike.

Stationarity deliberately still runs on the **raw** series: a unit root is a property of the
trend, not of a few spikes, and the differencing recommendation should describe the data the
caller actually has.

### 5. Test-suite defects

- **`test_stationarity` was collected by pytest as a test case.** The public API function shared
  the `test_*` prefix, so importing it into a test module made pytest try to run it, erroring
  with `fixture 'series' not found`. Renamed to `check_stationarity` across all call sites.
- **A shadowed test never ran.** `test_series_with_nans_handled` was defined twice in one file;
  the stationarity version at line 76 was silently overwritten. Renamed.
- **Two assertions could never pass.** The z-score NaN test used a 4-point series, where the
  largest attainable |z| is ~1.15 — below the 3σ threshold by construction; it now uses a
  200-point fixture. The conformal test asserted `α=0` returns a finite quantile, but α=0 demands
  100 % coverage and infinite width is the correct answer; the code was right and the test wrong.
- **Frequency detection had no dispersion check.** Deltas of 2 d, 7 d and 36 d have a median of
  exactly 7 d and were reported as clean weekly data. Now gated on the MAD/median ratio of the
  gaps — MAD rather than standard deviation, so a daily series that skips weekends still reads
  as daily.
- **Conformal split was off by one.** `int(0.7 * 350)` is 244, not 245, because 0.7 is not
  exactly representable. Now integer arithmetic.

### 6. The gate is on

- `ruff check`: 30 errors → 0 (17 autofixed; `StrEnum`, wrapped lines and unused imports by hand).
- CI runs `pytest tests/`, not `tests/unit/` — the 23 integration tests were previously excluded.
- `make lint` and the CI Lint step now run the identical command, so green locally means green
  in CI.

### 7. Dependencies

Core is the numeric stack only. FastAPI, LangGraph, torch, Celery, SQLAlchemy, wandb and Stripe
moved to `[api]`, `[agents]`, `[workers]`, `[db]`, `[viz]` and `[ops]`. A fresh `[dev,api]`
install now pulls **none** of the heavyweights — verified by inspecting the installed set.

Every dependency carries an upper bound. This is the direct lesson of the MAPIE failure: an
unpinned dependency silently removed a class and turned working code into dead code that still
imported cleanly. Ceilings make that loud at install time.

### 8. Config drift closed

Three files named three different, mutually inconsistent, non-existent entry points. All now
agree on `sibyl.api.app:create_app` (factory mode), and it is verified to boot.

| File | Was | Now |
|---|---|---|
| `Makefile` | `sibyl.api.main:app` | `sibyl.api.app:create_app --factory` |
| `Dockerfile` | `src.api.app:create_app` | same, plus `[api]` extra and `PYTHONPATH` |
| `docker-compose.yml` | `celery -A src.api.tasks` | worker removed until tasks exist |

Two changes beyond a rename: the compose `worker` ran a module that has never existed and could
only crash-loop — it is commented out with a note to restore it pointing at `sibyl.tasks`. And
`env_file` is now optional, since `.env` is gitignored and `docker compose up` previously failed
outright on a fresh clone.

### 9. README

Rewritten against reality: adds the forecasting layer and API, corrects the stage order, marks
each stack row *in use* or *planned* rather than implying all of it exists, and replaces the
"Getting started" flow that began with a command that could not succeed. The sample
`to_summary()` output is real generated output, not illustrative.

---

## What remains

**Not yet verified:**

- **`docker build` has never been run.** `docker compose config` validates and the uvicorn entry
  point was booted directly, but the image itself is unbuilt. The stub-package trick in the
  builder stage was updated for `src/forecasting`, and that path is untested.
- **CI has not yet run green.** Every check passes locally in a clean venv, but the first real
  Actions run is the proof. Local runs used Python 3.14; CI pins 3.11. The new pins admit both.

**Known limitations, unchanged:**

- `detected_periods` reports `[6, 7, 8, 13, 14]` for a pure period-7 signal — 6 and 8 are
  spectral leakage, 13/14 the harmonic. `dominant_period` is correct; the list is noisier than
  a caller would expect.
- `zscore_anomalies` casts the index with `int()`, so calling it directly on a
  `DatetimeIndex`-backed series will fail. The pipeline resets the index first, so this only
  affects direct use.
- Isolation Forest runs at a fixed `contamination=0.05`, so it flags ~5 % of points regardless
  of how many anomalies exist. In the 730-point fixture it flags 37 where 3 were injected. The
  de-spiking step tolerates this — interpolating a well-behaved point changes little — but the
  reported `anomaly_count` is closer to a fixed quota than a finding.

**Next up, in order:**

1. Push and confirm CI is green on Python 3.11.
2. `docker build` once, to close the last unverified path.
3. Then agents / tasks / persistence, each with its extra promoted as it lands.

---

## What is genuinely good here

- **The statistics are correct and well motivated.** Joint ADF+KPSS with opposite nulls, FFT
  peaks confirmed against an ACF Bartlett band, Shapiro-Wilk-gated selection between z-score and
  Isolation Forest — the right choices, made for the right stated reasons.
- **The conformal implementation is sound.** `_conformal_quantile` implements the exact
  finite-sample order statistic from Angelopoulos & Bates Theorem 1, correctly returns `inf`
  when the calibration set cannot bound the level rather than silently under-covering, and the
  empirical coverage tests at 80 % and 95 % pass. That is the hard part, and it is right.
- **The tests were doing their job the whole time.** They found four real bugs on their first
  ever execution. Nothing was listening — which is the entire lesson of this repository's first
  ten commits, and the reason step 1 was worth more than the seven that followed it.
