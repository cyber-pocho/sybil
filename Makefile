.PHONY: install test lint run-api run-worker migrate docker-build docker-up

install:
	pip install -e ".[dev,api,agents,db,workers]"
	@echo
	@echo "The agent layer is installed without a model provider, because it does"
	@echo "not have a default one. Add whichever you intend to use:"
	@echo "    pip install -e '.[anthropic]'   # or openai / google / mistral / groq / ollama"
	@echo "Then set SIBYL_LLM_PROVIDER and SIBYL_LLM_MODEL (see .env.example)."

test:
	pytest

# Matches the CI Lint step exactly — if this passes locally, CI passes.
# `ruff format` is deliberately not enforced: the source uses column-aligned
# assignments to make the maths line up, which the formatter would undo.
lint:
	ruff check src tests scripts alembic

run-api:
	uvicorn sibyl.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sibyl:latest .

docker-up:
	docker compose up -d

# The worker and the API must run the same schema, so migrate before either starts.
migrate:
	alembic upgrade head

run-worker:
	celery -A sibyl.tasks.celery_app worker --loglevel=info
