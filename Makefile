.PHONY: install test lint run-api docker-build docker-up

install:
	pip install -e ".[dev,api,agents]"

test:
	pytest

# Matches the CI Lint step exactly — if this passes locally, CI passes.
# `ruff format` is deliberately not enforced: the source uses column-aligned
# assignments to make the maths line up, which the formatter would undo.
lint:
	ruff check src tests

run-api:
	uvicorn sibyl.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sibyl:latest .

docker-up:
	docker compose up -d
