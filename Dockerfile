# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml .
# Stub the packages so setuptools' package discovery succeeds without the full
# source tree — this keeps the expensive dependency install in its own cached
# layer, independent of source edits. Must list every top-level package.
RUN mkdir -p src/sibyl src/diagnosis src/forecasting \
 && touch src/sibyl/__init__.py src/diagnosis/__init__.py src/forecasting/__init__.py

# The image runs both the API and the worker (same image, different
# command in compose), and both import the persistence layer.
RUN pip install --prefix=/install --no-cache-dir ".[api,db,workers]"


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Carry over everything pip installed in the builder
COPY --from=builder /install /usr/local

# ...but not the stub packages, which were only ever there so the dependency
# install could resolve. The real code arrives below via PYTHONPATH, and leaving
# empty namesakes in site-packages is how `forecasting` came to be shadowed: an
# empty *regular* package there beats a namespace package on PYTHONPATH no matter
# what the path order says, and the failure is an ImportError three layers deep.
RUN rm -rf /usr/local/lib/python3.11/site-packages/sibyl \
           /usr/local/lib/python3.11/site-packages/diagnosis \
           /usr/local/lib/python3.11/site-packages/forecasting

WORKDIR /app
COPY src/ ./src/
# Migrations ship with the image so `docker compose run api alembic upgrade head`
# works without mounting the repo.
COPY alembic/ ./alembic/
COPY alembic.ini .
# src/ holds the packages themselves, so it is the import root
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "sibyl.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
