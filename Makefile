# ── MK-RAG Project Makefile ───────────────────────────────────────────────────
# Usage:
#   make install        Install all dependencies
#   make download       Download all datasets
#   make setup          Process corpora and build indices
#   make test           Run all tests
#   make run-bm25       Run BM25 pipeline with gold dataset
#   make run-all        Run all 12 pipeline variants
#   make results        Print comparison table from saved results
#   make clean          Remove generated indices and processed data
#   make lint           Run ruff linter
#   make format         Format code with black

PYTHON     := python3
PIP        := $(PYTHON) -m pip
GOLD       := data/gold_dataset_50.jsonl
RESULTS    := results/

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install:
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: install
	$(PIP) install pytest pytest-cov ruff black ipython

.PHONY: download
download:
	$(PYTHON) scripts/download_datasets.py

.PHONY: download-wiki
download-wiki:
	$(PYTHON) scripts/download_datasets.py --only wiki

.PHONY: download-lvstck
download-lvstck:
	$(PYTHON) scripts/download_datasets.py --only lvstck

# ── Data pipeline ─────────────────────────────────────────────────────────────

.PHONY: setup
setup: setup-data build-indices

.PHONY: setup-data
setup-data:
	$(PYTHON) main.py setup-data

.PHONY: build-indices
build-indices:
	$(PYTHON) main.py build-indices --lang both

.PHONY: build-mk-index
build-mk-index:
	$(PYTHON) main.py build-indices --lang mk

# ── Tests ─────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

.PHONY: test-fast
test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -x  # stop on first failure

# ── Experiments ───────────────────────────────────────────────────────────────

.PHONY: run-bm25
run-bm25:
	$(PYTHON) main.py run-experiment \
		--pipeline mk_bm25 \
		--generator gpt4o \
		--gold-path $(GOLD) \
		--output-dir $(RESULTS)

.PHONY: run-dense
run-dense:
	$(PYTHON) main.py run-experiment \
		--pipeline mk_dense \
		--generator gpt4o \
		--gold-path $(GOLD) \
		--output-dir $(RESULTS)

.PHONY: run-hybrid
run-hybrid:
	$(PYTHON) main.py run-experiment \
		--pipeline mk_hybrid \
		--generator gpt4o \
		--gold-path $(GOLD) \
		--output-dir $(RESULTS)

.PHONY: run-all
run-all:
	$(PYTHON) main.py run-all \
		--gold-path $(GOLD) \
		--output-dir $(RESULTS)

.PHONY: results
results:
	$(PYTHON) main.py evaluate --results-dir $(RESULTS)

# ── Analysis ──────────────────────────────────────────────────────────────────

.PHONY: visualize
visualize:
	$(PYTHON) scripts/visualize_results.py --results-dir $(RESULTS)

.PHONY: latex-table
latex-table:
	$(PYTHON) scripts/visualize_results.py --results-dir $(RESULTS) --latex

# ── Code quality ──────────────────────────────────────────────────────────────

.PHONY: lint
lint:
	ruff check src/ tests/ scripts/ main.py

.PHONY: format
format:
	black src/ tests/ scripts/ main.py

.PHONY: check
check: lint test

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf data/processed/ data/indices/
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean-results
clean-results:
	rm -rf $(RESULTS)*.jsonl $(RESULTS)summary.json

.PHONY: help
help:
	@echo "MK-RAG Project Commands"
	@echo "========================"
	@echo "  make install          Install dependencies"
	@echo "  make download         Download all datasets"
	@echo "  make setup            Process data + build indices"
	@echo "  make test             Run all tests"
	@echo "  make run-bm25         Run BM25 pipeline"
	@echo "  make run-all          Run all 12 pipeline variants"
	@echo "  make results          Print comparison table"
	@echo "  make visualize        Generate result plots"
	@echo "  make latex-table      Generate LaTeX table for thesis"
	@echo "  make lint             Run linter"
	@echo "  make clean            Remove generated data/indices"
