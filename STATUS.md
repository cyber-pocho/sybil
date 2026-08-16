# Sibyl — Project Status Report

**Date:** 2026-08-16
**Branch:** `agent/provider-agnostic` @ `c56e45e`, pushed; not yet merged to `main`
**Basis:** full read-through, plus clean-room installs at three dependency levels,
lint, the test suite, Alembic against SQLite and real Postgres, a live agent run
against a local model, a real Redis/Celery/Postgres job round trip, a Docker build
and full `docker compose` probe, and the GitHub Actions history

---

## Verdict in one paragraph

**Diagnosis, forecasting, the HTTP API, the agent, the task queue and persistence
all work.** Vector search remains unstarted. The repository's defining problem —
a broken build backend that made `pip install -e .` fail, so **CI failed on all 10
commits and never executed a single test** — is fixed, and CI has since gone green
twice on Python 3.11. The suite went from *119 passed / 8 failed / 20 errors / 2
skipped* to **244 passed, 0 failed, 0 skipped**; `ruff check` from 30 errors to
clean. The agent is now provider-agnostic — it runs on any tool-calling chat model,
closed or open weights — and that change surfaced a second dependency-ceiling bug of
the same family as the original MAPIE one. The agent has now been run for real
against a local model, and the forecast job path against a real Redis, Celery worker
and Postgres. What remains unexercised is a live agent round-trip against any
*hosted* provider.

---

## Current state, verified

Everything below was executed, not inferred.

| Check | Result |
|---|---|
| `pip install -e ".[dev,api,agents,db,workers]"` (fresh venv) | ✅ exit 0 |
| `ruff check src/ tests/ scripts/ alembic/` | ✅ All checks passed |
| `pytest tests/` | ✅ **244 passed**, 0 failed, 0 skipped |
| Same suite with no provider package installed | ✅ 244 passed — the agent core carries no vendor dep |
| GitHub Actions on Python 3.11 | ✅ green — Install, Lint, Test all `success` |
| `docker compose config` | ✅ valid |
| `alembic upgrade head` → SQLite, then downgrade → upgrade | ✅ round-trips |
| `alembic upgrade head` → **real Postgres 16** | ✅ table + index as designed |
| Real Redis + real Celery worker + real Postgres round trip | ✅ 202 → queued → `done` with a forecast |
| Same, for a job that fails | ✅ recorded as `failed` with the reason; worker survived |
| Agent live against local `qwen2.5:7b` | ✅ ran; see "What the first live runs found" |
| `docker build` → `docker compose up` → full job round trip | ✅ 1.08 GB image; migrate, POST, worker, poll, `done` |

| Metric | Before (`0fb8648`) | Now |
|---|---|---|
| `pip install -e .` | ❌ `BackendUnavailable` | ✅ exit 0 |
| CI outcome | ❌ 10/10 runs failed at install | ✅ green |
| Tests executed in CI | **0, ever** | 244 |
| Test results | 119 pass / 8 fail / 20 error / 2 skip | **244 pass / 0 fail / 0 skip** |
| `ruff check` | 30 errors | **0** |
| Source LOC | 1,383 | 2,349 |
| Test LOC | 1,226 | 2,192 |
| LLM providers supported | 1 (hardcoded) | **7, none privileged** |

---

## Component inventory

| Component | LOC | Tests | Status |
|---|---|---|---|
| `diagnosis/profiler.py` | 169 | 20 | ✅ MAD dispersion gate — irregular spacing no longer reads as weekly |
| `diagnosis/stationarity.py` | 96 | 16 | ✅ renamed `test_stationarity` → `check_stationarity` |
| `diagnosis/seasonality.py` | 106 | ↑ | ✅ unchanged; now fed a de-spiked series by the pipeline |
| `diagnosis/anomalies.py` | 115 | 16 | ✅ unchanged |
| `diagnosis/pipeline.py` | 91 | 23 | ✅ reordered: profile → anomalies → stationarity → seasonality |
| `diagnosis/schemas.py` | 159 | — | ✅ `StrEnum`, long lines wrapped |
| `forecasting/base.py` | 65 | — | ✅ unchanged |
| `forecasting/prophet_model.py` | 143 | 22 | ✅ unchanged behaviour |
| `forecasting/ets_model.py` | 206 | 28 | ✅ **fixed** — was completely broken |
| `forecasting/conformal.py` | 133 | 22 | ✅ `ConformalWrapper` only; MAPIE removed |
| `sibyl/api/app.py` | 143 | 20 | ✅ `/health`, `/diagnose`, `/forecast` + polling |
| `sibyl/agents/forecaster_agent.py` | 252 | 21 | ✅ run live 8× against a local model |
| `sibyl/agents/llm.py` | 218 | 32 | ✅ provider registry; 7 providers, 3 constructed for real |
| `forecasting/registry.py` | 46 | — | ✅ shared by the agent and the worker |
| `sibyl/services/forecasting.py` | 93 | 12 | ✅ pure: no db, no queue, no HTTP |
| `sibyl/models/job.py` | 62 | — | ✅ one table, JSON params and result |
| `sibyl/db/{engine,base}.py` | 88 | — | ✅ sync engine, lazily built |
| `sibyl/tasks/{celery_app,forecast}.py` | 105 | 11 | ✅ verified against a real broker |
| Vector search | — | — | ⛔ not started; no package exists, only the `[vectorsearch]` extra |

---

## What was fixed

### 1. Build backend — the blocker behind everything else

`pyproject.toml` declared `build-backend = "setuptools.backends.legacy:build"`, a
module that does not exist. Now `setuptools.build_meta`.

That one line is why CI had failed on **every commit since the first**, always at
"Install dependencies", with Lint and Test skipped. It is also why the other bugs
survived: the tests that would have caught them were written, committed, and never
run.

### 2. ETS forecaster — was raising `KeyError` on every call

`ets_model.py` read `frame["mean_ci_lower"]` / `["mean_ci_upper"]` — the **SARIMAX**
column naming. `ETSResults.get_prediction().summary_frame()` returns `mean`,
`pi_lower`, `pi_upper`, and has since statsmodels 0.12, so this was never correct.
Fixed, clearing 19 errors and 1 failure. The duplicated `get_prediction()` call was
collapsed to one.

### 3. MAPIE — deleted

`MAPIEWrapper` imported `mapie.regression.MapieRegressor`, removed in MAPIE 1.x. The
`try/except ImportError` swallowed the failure, so its two tests silently skipped and
the class raised *"pip install mapie"* even with MAPIE installed. Removed the class,
`_PrefitStub`, its three tests and the dependency. `ConformalWrapper` already
implements the same split-conformal maths with verified coverage.

### 4. Outliers vs. seasonality — the one architectural fix

Three spikes in a 730-point series reduced seasonality detection to
`has_seasonality=False, strength=0.00` on a textbook weekly cycle: spike variance
dominates the FFT power budget and drags the ACF at lag 7 below the Bartlett band.
NaNs were never the problem — outliers were.

The pipeline detected those spikes at **step 4**, one step after they had already
ruined the step-3 measurement. Anomalies now run **second**, and seasonality runs on a
copy with flagged points interpolated — interpolated rather than dropped, because
dropping shifts every later observation and smears the periodicity being measured.

Measured end-to-end through `run_full_diagnosis`:

| Series | Before | After |
|---|---|---|
| clean | period 7, strength 0.88 | period 7, strength 0.88 |
| + 3 spikes | **none detected, 0.00** | **period 7, strength 0.89** |

The clean case is unchanged, which is the point: de-spiking costs nothing when there
is nothing to de-spike. Stationarity deliberately still runs on the **raw** series — a
unit root is a property of the trend, not of a few spikes.

### 5. Test-suite defects

- **`test_stationarity` was collected by pytest as a test.** The public API function
  shared the `test_*` prefix, so importing it made pytest try to run it. Renamed to
  `check_stationarity`.
- **A shadowed test never ran.** `test_series_with_nans_handled` was defined twice in
  one file; the stationarity version was silently overwritten. Renamed.
- **Two assertions could never pass.** The z-score NaN test used a 4-point series,
  where the largest attainable |z| is ~1.15 — below the 3σ threshold by construction.
  The conformal test asserted α=0 returns a finite quantile, but α=0 demands 100 %
  coverage and infinite width is correct; the code was right and the test wrong.
- **Frequency detection had no dispersion check.** Deltas of 2 d, 7 d and 36 d have a
  median of exactly 7 d and were reported as clean weekly data. Now gated on the
  MAD/median ratio — MAD rather than standard deviation, so a daily series that skips
  weekends still reads as daily.
- **Conformal split was off by one.** `int(0.7 * 350)` is 244, not 245, because 0.7 is
  not exactly representable. Now integer arithmetic.

### 6. The gate is on

`ruff check` 30 errors → 0. CI runs `pytest tests/`, not `tests/unit/` — the 23
integration tests were previously excluded. `make lint` and the CI Lint step now run
the identical command, so green locally means green in CI.

### 7. Dependencies

Core is the numeric stack only. FastAPI, LangGraph, torch, Celery, SQLAlchemy, wandb
and Stripe moved to `[api]`, `[agents]`, `[workers]`, `[db]`, `[viz]`, `[ops]` and
`[vectorsearch]`. Every dependency carries an upper bound — the direct lesson of the
MAPIE failure, where an unpinned dependency silently removed a class and turned
working code into dead code that still imported cleanly.

When the agent landed, `[agents]` was **split** rather than promoted as-is: it had
bundled FAISS, sentence-transformers and torch, none of which the agent imports.
Those moved to `[vectorsearch]` (unimplemented), so a `[dev,api,agents]` install pulls
none of them — verified against the installed set.

### 8. Version ceilings that had inverted — the MAPIE lesson, backwards

Making the agent provider-agnostic meant adding provider extras, and coherent extras
forced a look at the existing pins. All three were stale:

| Pin | Current release | Effect |
|---|---|---|
| `langgraph>=0.6,<1` | 1.2.11 | every 1.x release excluded |
| `langchain-core>=0.3,<1` | 1.5.5 | every 1.x release excluded |
| `langchain-anthropic>=0.3,<1` | 1.5.6 | every 1.x release excluded |

These predated LangChain's 1.0 and had quietly become the **opposite of what a ceiling
is for**. Instead of guarding against a breaking change, they were pinning the project
to an end-of-life major version and silently resolving to it on every fresh install.

This is the MAPIE failure mode inverted, and it is worth stating as a general rule: an
upper bound is a claim about the future that expires. MAPIE showed what happens with no
ceiling; this shows what happens when a ceiling is never revisited. Both end the same
way — installing something other than what the author believed was installed.

Not a judgement call: **the suite passes unmodified on LangChain 1.x**, checked before
touching the pins. All ceilings moved to `<2`, and every provider extra was dry-run
resolved. The `<1` pins would also have made the provider extras unsatisfiable —
`langchain-openai` 1.x requires `langchain-core` 1.x, which `langchain-core<1` forbids —
so this was load-bearing for the refactor, not incidental tidying.

### 9. Config drift closed

Three files named three different, mutually inconsistent, non-existent entry points.
All now agree on `sibyl.api.app:create_app` (factory mode), verified to boot both
under uvicorn and inside the container.

| File | Was | Now |
|---|---|---|
| `Makefile` | `sibyl.api.main:app` | `sibyl.api.app:create_app --factory` |
| `Dockerfile` | `src.api.app:create_app` | same, plus `[api,db,workers]` and `PYTHONPATH` |
| `docker-compose.yml` | `celery -A src.api.tasks` | `celery -A sibyl.tasks.celery_app worker` |

The compose `worker` ran a module that never existed and could only crash-loop, so it
was removed rather than left to fail. "Tasks and persistence" below gave it something
real to run, and it is back — on the same image as the API, because a separate image
would let the two drift, and a version skew between them shows up only as jobs that
fail once they reach the queue.

`env_file` is optional, since `.env` is gitignored and `docker compose up` previously
failed outright on a fresh clone. Postgres also gained a `pg_isready` healthcheck and
the API and worker now wait on it: Postgres accepts TCP connections a moment before it
will accept queries, and without the gate both raced it and crash-looped on first start.

### 10. `forecasting` was never actually packaged

Building and running the image — rather than trusting a green test suite — turned
up a packaging bug that had been latent since the layer was written.

`src/forecasting/` had no `__init__.py`. Its siblings `src/diagnosis/` and
`src/sibyl/` both do, so this was an inconsistency rather than a decision, and it
had two consequences:

- **`find_packages(where="src")` skipped it.** setuptools' `find_packages`
  requires `__init__.py`, so a non-editable `pip install sibyl` produced a
  distribution containing `diagnosis` and `sibyl` and **no `forecasting` at all**.
  Every install to date has been editable, which puts `src/` on the path directly
  and hides this completely.
- **The Docker image imported an empty stub instead of the real code.** The
  builder stage creates stub packages so dependencies can resolve without the
  source tree, and pip installed those stubs into site-packages. `/app/src` comes
  first on `sys.path`, but that does not help here: Python prefers a *regular*
  package found anywhere over a *namespace* package found earlier, so the empty
  `site-packages/forecasting/__init__.py` shadowed `/app/src/forecasting`.

It had never fired because nothing in the container imported `forecasting.*` —
`diagnosis` and `sibyl` are regular packages and resolved correctly. Importing
`forecasting.registry` from `api/app.py` was the first time anything asked.

Fixed by adding the missing `__init__.py`, which addresses both symptoms, and by
deleting the stub packages in the runtime stage so the shadowing cannot recur.
`src/sibyl/forecasting/` — an empty, git-tracked package left over from an earlier
layout — was removed at the same time.

The general lesson is the one this repository keeps relearning: a green suite says
the code works the way the tests import it. It said nothing about how the code
gets installed, and only `docker compose up` did.

---

## What was built

### HTTP API — `src/sibyl/api/app.py`

A `create_app()` factory with `/health` and `POST /diagnose`, which runs
`run_full_diagnosis` over row-oriented JSON records and returns the report plus the
six-line digest. Input the pipeline cannot diagnose returns 422 rather than 500. Ten
tests; verified serving under uvicorn and from the Docker container.

### Agent — `src/sibyl/agents/forecaster_agent.py`

A LangGraph ReAct loop (`START → agent ⇄ tools → END`) with three tools —
`diagnose`, `list_models`, `run_forecast`. The LLM reads the diagnosis, reads each
forecaster's `ModelCard`, picks a model, and explains the choice against what the
diagnosis found. Selection is the model's decision, not a rule table.

This is the consumer `BaseForecaster.card` was written for — its docstring already
said the card is *"returned by the agent when it explains its model selection to the
user"*. That consumer now exists.

Tools close over one series rather than taking it as an argument: a DataFrame cannot
survive a round trip through a JSON tool call. An unknown model name returns an error
string rather than raising, so the agent can correct itself instead of aborting the
graph; a `recursion_limit` bounds runaway loops.

**Eighteen tests, all against an injected stub LLM** — a real call would be
non-deterministic, slow, and would need some vendor's credential in CI, none of which
tests the graph. One of those tests initially passed for the wrong reason: the stub
replayed a single `AIMessage` object, and LangGraph's `add_messages` reducer merges by
id, so repeat turns silently overwrote instead of appending and the runaway loop could
not run away. The stub now mints a fresh id per reply.

### Provider registry — `src/sibyl/agents/llm.py`

The agent originally hardcoded `ChatAnthropic` and `claude-opus-5` inside `build_llm`.
That has been pulled out into a registry keyed by provider name, holding the only four
things that actually differ between vendors: the integration package, the class inside
it, the credential variable, and what that vendor calls the output cap. Seven providers
ship — anthropic, openai, google, mistral, groq for hosted models, and ollama plus
openai_compatible for open weights (vLLM, llama.cpp, LM Studio, TGI, OpenRouter).

Three design calls worth recording:

- **No default provider.** An unset `SIBYL_LLM_PROVIDER` raises. A default would make
  one vendor the quiet norm and every other one opt-in, which is the exact asymmetry
  the refactor removes — and it would fire a request at a paid endpoint nobody chose.
- **`temperature` is not in the signature.** It looks portable and is not: current
  Anthropic models reject it with a 400. It goes through `**kwargs` with every other
  vendor-specific knob, so the shared signature only contains genuinely shared things.
- **Open weights are not a special case.** `ollama` and `openai_compatible` reuse the
  same registry rows and the same code path as the hosted vendors. The only difference
  is `api_key_env=None` and a default `base_url`.

`_text_of` was also generalised. It stripped blocks of type `thinking` — one vendor's
spelling. It now strips `reasoning` and `reasoning_content` too, and falls back to any
non-reasoning block rather than returning an empty answer when no block self-identifies
as text.

**Thirty-two tests**, none of which construct a real client, so they run in CI with no
provider package installed. Separately and outside CI, `anthropic`, `ollama` and
`openai_compatible` were each constructed for real to confirm the per-vendor argument
names are right — that check is what proves `max_tokens` vs `max_output_tokens` vs
`num_predict` is correct rather than plausible.

One test here initially passed for the wrong reason, in the same family as the stub-id
bug above: it asserted that building an `ollama` model raises `ImportError`, which held
only because `langchain_ollama` happened to be absent. Installing the package broke it.
It now stubs the import and asserts the thing actually under test — that a self-hosted
provider clears the credential gate with no key set.

---

## What the first live runs found

Eight runs against `qwen2.5:7b` on a local Ollama daemon — free, no account, and the
first time any of this executed outside a stub. Two fixtures: **A**, 500 daily points
with a weekly cycle and 5 NaNs (prophet is correct); **B**, 60 monthly points, which
is below prophet's stated floor of 100 (ETS is correct).

The first run failed outright and usefully. Four distinct defects, all real:

| Defect | Fix | Where |
|---|---|---|
| Skipped `list_models` and `run_forecast`, answered anyway | Ordered steps made explicit; "you have not answered until `run_forecast` has run" | prompt |
| Chose **ARIMA**, which does not exist | `list_models` named as the only source of truth | prompt |
| Chose prophet on 60 points, quoting the 100 floor while ignoring it | Tool now **refuses** below the floor and names the models that fit | code |
| Reported a `[-inf, inf]` interval as a normal result | Tool now says the interval is infinite and why | code |

The split matters. The two prompt fixes ask a model to behave; the two code fixes
hold for every model regardless of capability, which is the right place for anything
load-bearing. The sample floor especially: it was advertised on every `ModelCard` and
enforced by nothing — `ProphetForecaster.fit()` will fit 60 points against its own
stated minimum of 100 without complaint. The card was documentation pretending to be
a contract.

**Result after the fixes**, over six further runs:

| Fixture | Tool calls needed | Outcome |
|---|---|---|
| A | 3 (diagnose → list_models → run_forecast) | correct, ~2 runs in 3 |
| B | 4 (the above, plus a retry after the refusal) | **0 in 6** |

Scenario B fails the same way every time, and not for want of understanding: the
refusal lands, the model reasons correctly to ETS, and then *describes* calling
`run_forecast` in prose instead of emitting a tool call. Making the refusal name the
eligible models and end in an imperative did not shift it. The conclusion is a
capability ceiling, not a prompt defect — **qwen2.5:7b cannot chain a fourth tool call
after a tool error** — and further prompt work against a 7B model is not the way to
close it. The recovery path itself is sound and deterministically covered by tests.

Worth recording for anyone choosing a model: the first attempt used
`qwen2.5-coder:7b`, which advertises `tools` capability in Ollama and does not
usefully have it — it emits `{"name": ..., "arguments": ...}` as message text with
`tool_calls=[]`, on a one-tool prompt with no other context. Declared capability is
not evidence; a two-line probe is.

### Tasks and persistence — `src/sibyl/{services,tasks,db,models}/`

`/diagnose` runs inline because a diagnosis is milliseconds. A Prophet fit is
seconds, so `/forecast` writes a `jobs` row, returns `202` with an id, and hands
the id to Celery. The worker runs it and writes the result back; the client polls
`GET /forecast/{id}`.

The layering is the point, and it is what makes this testable without infrastructure:

| Layer | Knows about |
|---|---|
| `services/forecasting.py` | records in, dict out. No database, no queue, no HTTP. |
| `tasks/forecast.py` | job rows and the service. Not HTTP. |
| `api/app.py` | HTTP and job rows. Never runs a forecast itself. |

`run_forecast_job` is a plain function; the Celery task is a three-line wrapper.
That is what lets the whole worker path be covered by tests that need no broker —
the only thing left untested is Celery's own delivery, which is not ours to test.
It was then verified separately against a real Redis, a real worker and a real
Postgres, in both the success and failure directions.

Decisions worth recording:

- **A failed forecast is a recorded outcome, not a crash.** The worker catches
  broadly and writes the message to the row. Letting an exception escape would
  leave the job stuck in `running` and a polling client waiting forever.
- **The forecast runs outside the database transaction.** Holding a connection
  open across a multi-second Stan fit buys nothing and costs a connection.
- **The task is dispatched after the transaction commits.** A worker can pick a
  message up before a slow commit lands, and would then look up a job that does
  not exist yet.
- **The queue carries only an id.** Everything else is in the row, so a retry
  reads current state rather than a snapshot from when the message was written.
- **`asyncpg` was replaced with `psycopg`.** The [db] extra shipped an async-only
  driver, which has no sync interface at all and so could never have backed the
  sync engine a Celery worker needs — a driver chosen for a design that was never
  built. It would have failed at the first real connection.
- **One migration, and a test that it stays honest.** Every other test creates
  tables with `create_all`, which would pass happily against a stale
  `alembic/versions`. `test_migration_matches_the_models` runs the real migration
  and diffs the result against `Base.metadata`, so a column added without a
  migration fails the suite rather than production.

The forecaster registry also moved out of the agent into
`forecasting/registry.py`. It was private to the agent, and the worker is a second
caller that must not import `sibyl.agents` to reach it — that module pulls in
LangGraph, and a worker fitting a Prophet model has no business requiring an LLM
stack.

`ruff` caught one real defect during this work: the new `/forecast` empty-body
test was named `test_empty_data_is_rejected`, which already existed for
`/diagnose`. It would have silently shadowed it — the same failure mode as the
duplicate `test_series_with_nans_handled` in §5, found this time by the linter
rather than by a full read-through.

---

## What remains

**Not yet verified:**

- **No hosted provider has been called.** The live verification below used a local
  `ollama` model only. Nothing has exercised Anthropic, OpenAI, Google, Mistral or
  Groq end to end, so their registry rows are verified as far as constructing a
  client and binding tools, and no further.
- **Scenario B has never completed on the model available.** See below — it is a
  model limit rather than a code one, but it means the tool-error recovery path is
  proven only against the stub, never live.

**Known limitations, unchanged:**

- `detected_periods` reports `[6, 7, 8, 13, 14]` for a pure period-7 signal — 6 and 8
  are spectral leakage, 13/14 the harmonic. `dominant_period` is correct; the list is
  noisier than a caller would expect.
- `zscore_anomalies` casts the index with `int()`, so calling it directly on a
  `DatetimeIndex`-backed series will fail. The pipeline resets the index first, so this
  only affects direct use.
- Isolation Forest runs at a fixed `contamination=0.05`, so it flags ~5 % of points
  regardless of how many anomalies exist — 37 where 3 were injected, in the 730-point
  fixture. The de-spiking step tolerates this, but the reported `anomaly_count` is
  closer to a fixed quota than a finding.
- The agent exposes only Prophet and ETS. Adding a forecaster to
  `forecasting/registry.py` is all it takes for both the agent and the worker to
  consider it, since the card supplies the description.
- **`run_full_diagnosis` raises on a perfectly noiseless linear series** —
  `ValueError: Invalid input, x is constant`, from a normality test handed
  zero-variance residuals after detrending. Found by accident while verifying the
  job failure path with a synthetic fixture that had no noise. Pre-existing and
  unrelated to persistence; confirmed by calling the pipeline directly. Real data
  always has noise, so this is a synthetic-input edge case rather than a live
  hazard — but the error names nothing a caller could act on.
- A job row stores the caller's full input in `params`, so a row is roughly as
  large as the series posted. Fine at this scale; the fix at a larger one is
  object storage plus a key, not a bigger column.

**Next up, in order:**

1. Run the agent against a hosted provider and compare with the local baseline
   above — in particular whether the 4-call recovery path works at all on a
   frontier model, which would confirm the ceiling is the model and not the design.
2. Decide whether the agent gets an endpoint. It is the one layer with no HTTP
   surface, deliberately: minutes per run, provider credentials in the worker, and
   the reliability profile measured above.
3. Fix the noiseless-linear-series crash listed above, or decide it is not worth it.
4. Then vector search, with `[vectorsearch]` promoted as it lands.

---

## What is genuinely good here

- **The statistics are correct and well motivated.** Joint ADF+KPSS with opposite
  nulls, FFT peaks confirmed against an ACF Bartlett band, Shapiro-Wilk-gated selection
  between z-score and Isolation Forest — the right choices, made for the right stated
  reasons.
- **The conformal implementation is sound.** `_conformal_quantile` implements the exact
  finite-sample order statistic from Angelopoulos & Bates Theorem 1, correctly returns
  `inf` when the calibration set cannot bound the level rather than silently
  under-covering, and the empirical coverage tests at 80 % and 95 % pass. That is the
  hard part, and it is right.
- **The architecture anticipated its own consumers.** `ModelCard` was designed before
  any agent existed, with a docstring naming exactly what would read it. When the agent
  was built, the interface it needed was already there and unchanged.
- **The agent's seam was in the right place already.** `run_agent(..., llm=...)` took
  an injected model from the start, purely so tests could pass a stub. That one
  parameter is why making the agent provider-agnostic touched a single function and no
  graph code: the abstraction the refactor needed had been sitting there since the
  layer was written, for an unrelated reason.
- **The tests were doing their job the whole time.** They found four real bugs on their
  first ever execution. Nothing was listening — which is the entire lesson of this
  repository's first ten commits, and the reason fixing one line in `pyproject.toml`
  was worth more than the seven steps that followed it.
