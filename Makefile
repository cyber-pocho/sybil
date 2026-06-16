.PHONY: install test lint run-api docker-build docker-up

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

run-api:
	uvicorn sibyl.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sibyl:latest .

docker-up:
	docker compose up -d
