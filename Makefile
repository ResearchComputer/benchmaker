# benchmaker — common dev / release tasks (uv-based).
# Override tools if needed:  make UV=uv PYTHON=.venv/bin/python <target>

UV     ?= uv
PYTHON ?= .venv/bin/python
TWINE  ?= uvx twine

.DEFAULT_GOAL := help

.PHONY: help install install-dev test clean build check \
        publish-test publish version

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Editable install (library + CLI)
	$(UV) pip install -e .

install-dev: ## Editable install with dev + hf + rich extras
	$(UV) pip install -e ".[dev,hf,rich]"

test: ## Run the test suite
	$(UV) run pytest -q

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info bench_maker.egg-info benchmaker.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

version: ## Print pyproject vs package __version__ (they must match before release)
	@echo "pyproject.toml : $$(grep -m1 '^version' pyproject.toml | sed -E 's/.*\"(.*)\".*/\1/')"
	@echo "__init__.py    : $$($(PYTHON) -c 'import benchmaker; print(benchmaker.__version__)')"

build: clean ## Build sdist + wheel into dist/
	$(UV) build

check: build ## Validate built artifacts with twine
	$(TWINE) check dist/*

# --- Publishing ---------------------------------------------------------------
# Uploading to a package index is public and irreversible (a version can't be
# re-uploaded once it exists). Bump the version in BOTH pyproject.toml and
# benchmaker/__init__.py first (see `make version`), then dry-run on TestPyPI.
# Credentials come from ~/.pypirc (or TWINE_USERNAME/TWINE_PASSWORD). Never
# hard-code tokens here.

publish-test: check ## Upload to TestPyPI (dry run for the real thing)
	$(TWINE) upload --repository testpypi dist/*

publish: check ## Upload to PyPI (public, irreversible)
	$(TWINE) upload dist/*
