#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sys_des_project
PYTHON_VERSION = 3.10
VERSION ?= v1

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.DEFAULT_GOAL := help

## Install environment from scratch (uv + deps + dev deps)
.PHONY: setup
setup:
	uv sync --all-groups

## Run all checks (no fixes): style + types
.PHONY: check
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy .

## Lint only (no fixes)
.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .

## Format code (mutates files)
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .

## Run tests
.PHONY: test
test:
	uv run pytest tests

## Delete compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Start local S3 (RustFS + init bucket)
.PHONY: s3-up
s3-up:
	docker compose up -d

## Stop local S3
.PHONY: s3-down
s3-down:
	docker compose down

## Ingest raw CIFAR-10 archive into S3 (idempotent)
.PHONY: ingest
ingest:
	uv run python -m sys_des_project.ingest

## Run full data pipeline (download -> convert -> split -> stats -> upload)
.PHONY: data
data:
	uv run python -m sys_des_project.cli download
	uv run python -m sys_des_project.cli convert
	uv run python -m sys_des_project.cli split --val-size 5000 --seed 42
	uv run python -m sys_des_project.cli stats
	uv run python -m sys_des_project.cli upload --version $(VERSION)

## Start full infra: S3 + Postgres + MLflow + образ для обучения
.PHONY: infra-up
infra-up:
	mkdir -p runs
	docker compose up -d --build
	docker compose build train

## Stop full infra
.PHONY: infra-down
infra-down:
	docker compose down

## Run one experiment in a container (CONFIG=configs/simple_cnn.yaml)
.PHONY: train
train:
	GIT_COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo unknown) \
		docker compose run --rm -T train python -m sys_des_project.train --config $(CONFIG)

## Run one experiment on the host, without Docker (debug)
.PHONY: train-local
train-local:
	uv run python -m sys_des_project.train --config $(CONFIG)

## Run a grid; каждый прогон — в отдельном контейнере (CONFIG=configs/grid.yaml)
.PHONY: experiments
experiments:
	GIT_COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo unknown) \
	GRID_RUNNER=docker uv run python -m sys_des_project.grid --config $(CONFIG)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)