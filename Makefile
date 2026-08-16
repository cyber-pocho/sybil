.PHONY: install test lint run-api docker-build docker-up

install:
	pip install -e ".[dev,api,agents]"
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
	ruff check src tests

run-api:
	uvicorn sibyl.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sibyl:latest .

docker-up:
	docker compose up -d
