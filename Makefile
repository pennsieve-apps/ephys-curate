.PHONY: help models test fixture run clean

SERVICE_NAME ?= ephys-curate
PYTHON ?= python3

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make models   - download UnitRefine's pretrained classifiers (pinned, sha256-checked) to models/"
	@echo "make test     - pytest: the synthetic fixture through the processor, both layouts, Kilosort input (no Docker)"
	@echo "make fixture  - write the synthetic session (8 good, 1 contaminated, 3 noise units) to data/fixture/{linked,flat}/"
	@echo "make run      - build the image and run it in Docker over data/fixture/linked/, writing to data/output/"
	@echo "make clean    - remove data/output/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

models:
	$(PYTHON) scripts/fetch_models.py models

test: models
	UNITREFINE_MODELS_DIR=models $(PYTHON) -m pytest -q tests

fixture:
	rm -rf data/fixture
	$(PYTHON) -m tests.make_fixture data/fixture --layout both

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -maxdepth 2 | sort

clean:
	rm -rf data/output
