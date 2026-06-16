# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml .
# Stub the packages so setuptools find-packages succeeds without the full source tree
RUN mkdir -p src/sibyl src/diagnosis \
 && touch src/sibyl/__init__.py src/diagnosis/__init__.py

RUN pip install --prefix=/install --no-cache-dir .


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Carry over everything pip installed in the builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
