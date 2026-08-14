# Sibyl — Project Status Report

> **⚠️ Historical. This is the audit taken at `0fb8648`, before the remediation pass.**
> Every item below (B0–B5, T1–T4, P3, P4, P5) has since been fixed. Current state:
> `pip install -e ".[dev,api]"` works, `ruff check` is clean, and **157 tests pass**
> with 0 failures and 0 skips. Kept as the record of what was wrong and why — see
> `README.md` for how the project actually works now.

**Date:** 2026-08-14
**Branch:** `main` @ `0fb8648` (working tree clean)
**Reviewer:** automated read-through + clean-room build and test run

---

## Verdict in one paragraph

The **diagnosis layer is real and mostly works**. The **forecasting layer is half-working**:
Prophet and the hand-rolled conformal wrapper pass their tests; the ETS forecaster is broken
and the MAPIE wrapper is dead code. The **application layer (`src/sibyl/`) does not exist** —
it is seven empty `__init__.py` files, despite the README describing a FastAPI + LangGraph +
Celery + Postgres product. Most importantly: **`pip install -e .` has never worked**, so CI has
failed on every single one of the 10 commits in this repo's history and **not one test has ever
executed in CI**. The 8 test failures + 20 errors documented below are the accumulated cost of
that.

---

## How this was verified

Nothing here is inferred from reading alone. I:

1. Reproduced the packaging failure in a clean venv (`pip install -e .` → `BackendUnavailable`).
2. Patched the build backend in a scratch copy, installed the real dependencies, and ran the
   full suite.
3. Pulled the GitHub Actions history via `gh` to confirm every run failed at the same step.

**Environment caveat:** the test run used Python 3.14 / pandas 3.0.5 / statsmodels 0.14.6 /
prophet 1.3.0, whereas CI targets Python 3.11. I traced each failure to its root cause and
confirmed every one is a logic or expectation error that is **independent of the Python/pandas
version** — none of them are "new pandas broke it" artifacts.

---

## Component inventory

| Component | Files | Status |
|---|---|---|
| `diagnosis/profiler.py` | 152 L | ✅ Works. 1 real edge-case gap (see B4). |
| `diagnosis/stationarity.py` | 97 L | ✅ Works. ADF+KPSS logic is sound. |
| `diagnosis/seasonality.py` | 106 L | ⚠️ Works on clean data, **collapses on outliers** (B3). |
| `diagnosis/anomalies.py` | 115 L | ✅ Works. |
| `diagnosis/pipeline.py` | 70 L | ⚠️ Works, but stage ordering causes B3. |
| `diagnosis/schemas.py` | 147 L | ✅ Complete, incl. `to_summary()` LLM digest. |
| `forecasting/base.py` | 65 L | ✅ Clean ABC contract. |
| `forecasting/prophet_model.py` | 139 L | ✅ **22/22 tests pass.** |
| `forecasting/ets_model.py` | 197 L | ❌ **`predict()` raises `KeyError` — completely broken** (B1). |
| `forecasting/conformal.py` → `ConformalWrapper` | ~120 L | ✅ Works; empirical coverage tests pass. One off-by-one (B5). |
| `forecasting/conformal.py` → `MAPIEWrapper` | ~100 L | ❌ **Dead code against current MAPIE** (B2). |
| `sibyl/api`, `agents`, `services`, `tasks`, `db`, `models` | 0 L each | ⛔ **Not started.** Empty packages. |

**Code volume:** ~1,013 lines of source, ~1,596 lines of tests. Test-to-source ratio is high —
the discipline is there, the tests just never ran.

---

## P0 — The blocker that caused everything else

### B0. Invalid PEP 517 build backend → install fails → CI has never run a test

`pyproject.toml:3`

```toml
build-backend = "setuptools.backends.legacy:build"   # this module does not exist
```

There is no `setuptools.backends` module. The correct value is `setuptools.build_meta`
(or `setuptools.build_meta:__legacy__`). Reproduced:

```
pip._vendor.pyproject_hooks._impl.BackendUnavailable:
    Cannot import 'setuptools.backends.legacy'
```

**Blast radius:**
- `make install` fails locally.
- `docker build` fails at `RUN pip install --prefix=/install .` (Dockerfile:17).
- CI fails at "Install dependencies"; **Lint and Test are skipped every time.**

**CI history — 10 runs, 10 failures, all at the same step:**

| Run | Commit | Result |
|---|---|---|
| 27709967237 | added: srcs on forecasting layer | ❌ Install deps |
| 27697797686 | added forecaster | ❌ Install deps |
| 27635784201 | added: Forecasting srcs | ❌ Install deps |
| 27635015181 | added: main diagnosis pipeline | ❌ Install deps |
| 27634436180 | added: README.md | ❌ Install deps |
| …5 more | …back to the first commit | ❌ Install deps |

**Fix:** one line. This is the highest-leverage change in the repo — it turns the test suite
from decoration into a safety net.

---

## P1 — Confirmed bugs (each reproduced by running the code)

### B1. `ETSForecaster.predict()` is completely broken — wrong statsmodels column names

`src/forecasting/ets_model.py:143-158`

The code reads `frame_80["mean_ci_lower"]` / `["mean_ci_upper"]`. Verified against statsmodels
0.14.6, `ETSResults.get_prediction().summary_frame()` returns:

```python
['mean', 'pi_lower', 'pi_upper']
```

`mean_ci_*` is the **SARIMAX** naming, not the ETS naming — this has been true since
statsmodels 0.12, so it was never correct, it just never ran. Every call to `predict()`
raises `KeyError: 'mean_ci_lower'`.

**Impact: 19 errors + 1 failure — the entire ETS test module.** Fix is a rename to
`pi_lower`/`pi_upper`.

*While in there:* lines 139-140 call `get_prediction(start=n, end=n+horizon-1)` twice with
identical arguments and assign to `pred_80`/`pred_95`. The alpha is applied later in
`summary_frame()`, so one call suffices — the second is pure waste.

### B2. `MAPIEWrapper` is dead code against any current MAPIE

`src/forecasting/conformal.py:31-36`

```python
from mapie.regression import MapieRegressor   # removed in MAPIE 1.x
```

Verified with MAPIE 1.5.0 (the version `pip install mapie` gives you today):

```
ImportError: cannot import name 'MapieRegressor' from 'mapie.regression'
```

The current API is `SplitConformalRegressor`, `CrossConformalRegressor`,
`JackknifeAfterBootstrapRegressor`, and — most relevant here — **`TimeSeriesRegressor`**.

This failure mode is especially nasty because the `try/except ImportError` swallows it:
`_MAPIE_AVAILABLE` silently becomes `False`, the two MAPIE tests silently **skip**, and
`MAPIEWrapper(...)` raises *"requires MAPIE: pip install mapie"* even when MAPIE is installed.
~100 lines of code and 2 tests that look green and are actually inert.

**Decision needed:** port to `SplitConformalRegressor`/`TimeSeriesRegressor`, or delete it.
`ConformalWrapper` already works and its coverage guarantees are verified — MAPIE earns its
place only if you want Jackknife+ / CV+ variants.

### B3. Three outliers destroy seasonality detection — and the pipeline ordering guarantees it

`src/diagnosis/seasonality.py` + `src/diagnosis/pipeline.py:57-62`

The integration test builds a 730-point daily series with a strong weekly cycle, injects
3 spikes at `500.0`, and asserts weekly seasonality is found. It isn't. I isolated the cause:

| Series variant | Result |
|---|---|
| clean | `periods=[6,7,8,13,14]`, dominant=**7**, strength=**0.91** ✅ |
| + 5 NaNs | `periods=[6,7,8,13,14]`, dominant=**7**, strength=**0.90** ✅ |
| + 3 spikes | `has_seasonality=False`, strength=**0.00** ❌ |
| + spikes + NaNs | `has_seasonality=False`, strength=**0.00** ❌ |

NaNs are handled fine. **Outliers are the killer**: spike variance dominates the FFT power
budget and drags the ACF at lag 7 below the Bartlett ±2/√n band, so the confirmation step
rejects the true period.

This is architectural, not a typo. `run_full_diagnosis` runs seasonality at **step 3** and
anomalies at **step 4** — the pipeline finds the spikes one step *after* they have already
poisoned the seasonality result, and never feeds that knowledge back.

**Options:** (a) reorder to detect anomalies first and pass a winsorized series into
seasonality; (b) make seasonality robust internally (median/MAD clipping before the FFT);
(c) run seasonality twice — once raw, once on the cleaned series — and report both.
Option (a) is the smallest change and matches the "each step builds on the last" intent
already stated in the pipeline docstring.

*Related quality note:* even on clean data the detector reports `[6, 7, 8, 13, 14]` for a pure
period-7 signal. 6 and 8 are spectral leakage around the true peak, 13/14 are the harmonic.
`dominant_period` is correct, but `detected_periods` is noisier than a caller would expect.

### B4. Frequency detection has no dispersion check

`src/diagnosis/profiler.py:63-73`

`_detect_frequency` uses only the **median** inter-row delta. For timestamps spaced 2 d, 7 d,
36 d apart the median is exactly 7 days → classified `weekly`, when the data is plainly
irregular. `test_irregular_frequency` fails on precisely this case.

The test's expectation is the right one. A spread check — e.g. flag `irregular` when
IQR/median or MAD/median exceeds a threshold — would fix it without disturbing the well-behaved
cases.

### B5. Float truncation makes the conformal train/cal split off by one

`src/forecasting/conformal.py:75` (and the identical line at `:169`)

```python
n_train = int(0.7 * n)
```

For `n=350`: `0.7 * 350 == 244.99999999999997` → `int()` → 244, so the calibration set is
**106** points, not the intended 105. `test_metadata_has_n_calibration` catches it.

Harmless statistically, but it makes the split silently non-reproducible across `n`.
Use `round(0.7 * n)` or integer arithmetic (`7 * n // 10`).

---

## P2 — Test-suite defects

### T1. The library function `test_stationarity` is collected by pytest as a test

`src/diagnosis/stationarity.py:31` — the public API is named `test_stationarity()`. Because
`tests/unit/test_stationarity_seasonality.py` imports it into module scope, **pytest collects
it as a test case** and errors with `fixture 'series' not found`.

This is a naming collision baked into the public API, so it will recur in every module that
imports it. Cleanest fix: rename to `check_stationarity()` (or `assess_stationarity()`) and
update the four call sites. Cheap fix: `from ... import test_stationarity as _test_stationarity`.

### T2. A test is silently shadowed and never runs

`tests/unit/test_stationarity_seasonality.py:76` and `:123` both define
`test_series_with_nans_handled`. The second definition wins; **the stationarity NaN test at
line 76 never executes.** (ruff flags this as F811.)

### T3. Two tests assert things that can never be true

- `tests/unit/test_anomalies.py:56` — `zscore_anomalies(pd.Series([1.0, nan, 2.0, 100.0]))`
  expects `100.0` to be flagged. After `dropna()` there are 3 points, and with n=3 the maximum
  attainable |z| is ~1.15 — **below the 3σ threshold by construction.** The test can never
  pass; it needs a longer series.
- `tests/unit/test_conformal.py:71` — asserts `_conformal_quantile(scores, 5, alpha=0.0) == 5.0`.
  The comment's arithmetic is wrong: `ceil(1.0 × 6) = 6 > 5`, so `inf` is returned. **The code
  is right and the test is wrong** — α=0 legitimately demands infinite width.

### T4. CI would not run the integration tests even if it installed

`.github/workflows/ci.yml:25` runs `pytest tests/unit/ -v`. `tests/integration/` (174 lines,
the only end-to-end coverage of `run_full_diagnosis`) is excluded. Note `pyproject.toml`
already sets `testpaths = ["tests"]`, so plain `pytest` picks up both — the explicit path in
CI is what narrows it.

---

## P3 — Lint and formatting

Ruff has never run in CI. Current state:

- **`ruff check src/ tests/` → 30 errors** (15 auto-fixable)
  - 4× `F401` unused imports — incl. `BaseEstimator`/`RegressorMixin` in `conformal.py:33` and
    `numpy` in `stationarity.py:22`
  - 1× `F811` redefinition (see T2 — a real bug, not style)
  - 11× `UP045` `Optional[X]` → `X | None`
  - 1× `UP042` `class DetectedFrequency(str, Enum)` → `StrEnum`
  - 10× `E501` lines over 100 chars
  - 3× `I001` unsorted import blocks
- **`ruff format --check src tests` → 18 of 30 files would be reformatted**

Note that `make lint` runs `ruff format --check` but **CI does not** — so the Makefile and the
workflow disagree about what "lint" means. Worth reconciling before you turn the gate on, or
the first green build will be a surprise.

---

## P4 — Configuration drift

Three files name three different, mutually inconsistent, **and non-existent** application
entry points:

| File | Declares | Reality |
|---|---|---|
| `Makefile:14` | `uvicorn sibyl.api.main:app` | `src/sibyl/api/` is empty |
| `Dockerfile:35` | `uvicorn src.api.app:create_app --factory` | no `src/api/`, no `create_app` |
| `docker-compose.yml:13` | `celery -A src.api.tasks worker` | no `src/api/tasks` |

None of these can start today. They need to converge on one path once the API actually exists.

**Other drift:**

- **No dependency pinning at all.** Every entry in `pyproject.toml` is unbounded. That is
  exactly how B2 (MAPIE 1.x API removal) slipped in, and it makes the build
  non-reproducible — my clean install pulled pandas **3.0.5**, a major version the code has
  never been tested against. At minimum pin the numeric stack (`pandas`, `numpy`,
  `statsmodels`, `scikit-learn`, `mapie`, `prophet`).
- **Dependency footprint vs. reality.** `torch`, `wandb`, `stripe`, `faiss-cpu`,
  `sentence-transformers`, `langgraph`, `langchain*`, `sqlalchemy`, `asyncpg`, `alembic`,
  `celery` are all hard runtime requirements, and **not one of them is imported anywhere in
  the codebase.** They make every install slow and every version conflict more likely, for
  zero present benefit. Moving them to optional extras until their layer is built would cut
  install time dramatically.
- **Namespace confusion:** there is a top-level `src/forecasting/` (real code) *and* an empty
  `src/sibyl/forecasting/`. Pick one home before the app layer starts importing.
- `src/forecasting/` has no `__init__.py` while every sibling package does. Modern setuptools
  packages it as a namespace package and the wheel comes out correct (I verified the built
  wheel contains all five modules), so this is **cosmetic, not broken** — but the
  inconsistency is worth closing.

---

## P5 — Documentation drift

`README.md` is well written and now describes a smaller project than exists in some places and
a much larger one in others:

- **Missing from the layout tree:** `src/forecasting/` entirely (5 modules, 696 lines),
  `diagnosis/pipeline.py`, and `tests/integration/`.
- **Overstates the app layer:** the Stack table lists API, Agents, Vector search, Task queue,
  Database, Observability and Payments as if implemented. All seven are empty directories.
- **Environment variables** (`ANTHROPIC_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `WANDB_API_KEY`,
  `STRIPE_SECRET_KEY`, `FAISS_INDEX_PATH`) are documented as "required before running" but
  nothing in the codebase reads any of them.
- **"Getting started" cannot be followed** — step one is `make install`, which fails (B0).

---

## Recommended order of work

**Sequenced so each step makes the next one verifiable.**

1. **Fix `build-backend`** (`pyproject.toml:3` → `setuptools.build_meta`). One line. This is
   the gate on everything below — until it lands, no automated check in this repo does
   anything.
2. **Green the suite.** In dependency order: B1 (ETS column names, clears 20 of the 28
   failures), then T1/T2/T3 (test defects), then B5 (split off-by-one).
3. **Decide MAPIE's fate** (B2): port to `SplitConformalRegressor`/`TimeSeriesRegressor`, or
   delete `MAPIEWrapper` and its two skipped tests. Do not leave it silently inert.
4. **Fix B3 (outliers vs. seasonality)** — the only genuinely architectural item. Reordering
   the pipeline to clean before it characterises is the smallest change that honours the
   design already described in the docstring.
5. **Turn on the gate.** Add `ruff --fix`, reconcile `make lint` with the CI step, and widen CI
   to `pytest tests/` so the integration tests actually run.
6. **Pin the numeric stack**, and demote the unused heavyweight dependencies to extras.
7. **Then** start `src/sibyl/` — with a single agreed entry point propagated to Makefile,
   Dockerfile, and docker-compose.
8. **Refresh the README** to match reality; mark the app-layer stack as planned rather than
   present.

---

## What is genuinely good here

Worth saying plainly, because the failure list above is long and the underlying work is not bad:

- **The statistics are correct and well motivated.** Joint ADF+KPSS with opposite nulls, FFT
  peaks confirmed against an ACF Bartlett band, Shapiro-Wilk-gated method selection between
  z-score and Isolation Forest — these are the right choices, made for the right stated
  reasons.
- **The conformal implementation is sound.** `_conformal_quantile` implements the exact
  finite-sample order statistic from Angelopoulos & Bates Theorem 1, correctly returns `inf`
  when the calibration set is too small to bound the level rather than silently under-covering,
  and the **empirical coverage tests at 80% and 95% pass.** That is the hard part of conformal
  prediction and it is right.
- **Prophet integration: 22/22 passing**, including interval ordering and monotonicity.
- **Test coverage intent is strong** — 1,596 lines of tests against 1,013 of source, with real
  statistical assertions rather than smoke tests. The tests found B1, B3, B4 and B5 *on their
  first ever execution*. The suite was doing its job the whole time; nothing was listening.
- **The code matches its stated style** — explicit math, comments that explain *why*, no
  premature abstraction. It reads like the research notebook `CLAUDE.md` asks for.

The gap between this project's quality and its status is almost entirely one broken line in
`pyproject.toml`.
