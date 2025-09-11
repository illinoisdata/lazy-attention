.ONESHELL:
ENV_PREFIX=$(shell python3 -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
USING_POETRY=$(shell grep "tool.poetry" pyproject.toml && echo "yes")
SOURCE_DIRS=drag/ benchmarks/ tests/
MAX_LINE_LENGTH=127

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo " "
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:             ## Show the current environment.
	@echo "Current environment:"
	@if [ "$(USING_POETRY)" ]; then poetry env info && exit; fi
	@echo "Running using $(ENV_PREFIX)"
	@$(ENV_PREFIX)python3 -V
	@$(ENV_PREFIX)python3 -m site

.PHONY: virtualenv
virtualenv:       ## Create a virtual environment.
	@if [ "$(USING_POETRY)" ]; then poetry install && exit; fi
	@echo "creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo "creating virtualenv for cacheblend ..."
	@rm -rf .venv-cacheblend
	@python3 -m venv .venv-cacheblend
	@./.venv-cacheblend/bin/pip install -U pip
	@echo
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"
	@echo "!!! Please run 'source .venv-cacheblend/bin/activate' to enable the environment (for cacheblend) !!!"

.PHONY: install
install:          ## Install the project in dev mode.
	@if [ "$(USING_POETRY)" ]; then poetry install && exit; fi
	$(ENV_PREFIX)pip install wheel setuptools  # Need before installing xformers
	$(ENV_PREFIX)pip install torch torchvision torchaudio  # Need before installing xformers
	$(ENV_PREFIX)pip install xformers --no-dependencies
	$(ENV_PREFIX)pip install -e .[dev]
	(cd promptcache/ && pip install -e .)

.PHONY: installcacheblend
installcacheblend: install          ## Install cacheblend baseline.
	(cd CacheBlend/vllm_blend/ && pip install -e .)
	(cd CacheBlend/ && pip install -r requirements.txt)

.PHONY: fmt
fmt:              ## Format code using black & isort.
	$(ENV_PREFIX)isort ${SOURCE_DIRS}
	$(ENV_PREFIX)black -l ${MAX_LINE_LENGTH} ${SOURCE_DIRS}

.PHONY: lint
lint:             ## Run pep8, black, mypy linters.
	$(ENV_PREFIX)flake8 --select=E9,F63,F7,F82 --show-source ${SOURCE_DIRS}
	$(ENV_PREFIX)flake8 --max-complexity=100 --max-line-length=${MAX_LINE_LENGTH} --extend-ignore=W503,E203,E701 ${SOURCE_DIRS}
	$(ENV_PREFIX)black -l ${MAX_LINE_LENGTH} --check ${SOURCE_DIRS}
	$(ENV_PREFIX)mypy --config-file pyproject.toml --ignore-missing-imports ${SOURCE_DIRS}

.PHONY: test
test: lint        ## Run tests and generate coverage report.
	$(ENV_PREFIX)pytest -vv --cov-config .coveragerc --cov=drag -l --tb=short --maxfail=1 tests
	$(ENV_PREFIX)coverage xml
	$(ENV_PREFIX)coverage html

.PHONY: clean
clean:            ## Clean unused files.
	@find ${SOURCE_DIRS} -name '*.pyc' -exec rm -f {} \;
	@find ${SOURCE_DIRS} -d -name '__pycache__' -exec rm -rf {} \;
	@find ${SOURCE_DIRS} -name 'Thumbs.db' -exec rm -f {} \;
	@find ${SOURCE_DIRS} -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
