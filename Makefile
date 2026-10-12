#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sys_des_project
PYTHON_VERSION = 3.10

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

## Make dataset
.PHONY: data
data: setup
	uv run python sys_des_project/dataset.py

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