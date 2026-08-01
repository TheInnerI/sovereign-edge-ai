.PHONY: setup test-synthetic test-model stats clean train-embedder phone-guide

# ─── Setup ────────────────────────────────────────────────────────────

VENV := .venv
PYTHON := $(VENV)/bin/python

$(VENV):
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -r requirements.txt

setup: $(VENV)
	@echo "✅ Virtual environment ready"

# ─── Embedder ─────────────────────────────────────────────────────────

train-embedder: $(VENV)
	$(PYTHON) run.py train-embedder

# ─── Model ────────────────────────────────────────────────────────────

MODEL_DIR := models
MODEL_FILE := $(MODEL_DIR)/Bonsai-8B-Q1_0.gguf

download-model: $(VENV)
	@echo "📥 Downloading Bonsai-8B 1-bit (1.15 GB)..."
	$(PYTHON) -c "\
	from huggingface_hub import hf_hub_download; \
	path = hf_hub_download('prism-ml/Bonsai-8B-gguf', 'Bonsai-8B-Q1_0.gguf', local_dir='$(MODEL_DIR)', local_dir_use_symlinks=False); \
	print(f'Downloaded: {path}')"

$(MODEL_FILE):
	$(MAKE) download-model

# ─── Ollama Import ────────────────────────────────────────────────────

ollama-import: $(MODEL_FILE)
	@echo "📦 Importing into Ollama..."
	cd $(CURDIR) && ollama create sovereign-observer -f Modelfile
	@echo "✅ Model imported. Run: ollama run sovereign-observer"

# ─── Testing ──────────────────────────────────────────────────────────

test-synthetic: $(VENV) train-embedder
	@echo "🧪 Running synthetic tests (no model needed)..."
	$(PYTHON) tests/test_synthetic.py

test-model: $(VENV) $(MODEL_FILE)
	@echo "🧪 Running model inference tests..."
	$(PYTHON) run.py test-model

test: test-synthetic
	@echo "✅ All tests passed"

# ─── CLI ──────────────────────────────────────────────────────────────

observe: $(VENV)
	$(PYTHON) run.py observe --intent "$(INTENT)" --predicted "$(PREDICTED)" --executed "$(EXECUTED)" --actual "$(ACTUAL)"

interactive: $(VENV)
	$(PYTHON) run.py interactive

stats: $(VENV)
	$(PYTHON) run.py stats

# ─── Phone ────────────────────────────────────────────────────────────

phone-guide:
	@cat PHONE-GUIDE.md

# ─── Clean ────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) __pycache__ src/__pycache__ src/*/__pycache__
	@echo "🧹 Cleaned"

clean-all: clean
	rm -rf data/ models/ .venv/
	@echo "🧹 Full clean (data + models + venv removed)"

# ─── Help ─────────────────────────────────────────────────────────────

help:
	@echo "Sovereign Edge Observer — Makefile"
	@echo ""
	@echo "  make setup              Create virtual environment"
	@echo "  make train-embedder     Fit TF-IDF+SVD embedder"
	@echo "  make download-model     Download Bonsai-8B-Q1_0.gguf (1.15 GB)"
	@echo "  make ollama-import      Import model into Ollama"
	@echo "  make test-synthetic     Run 10 synthetic tests (fast, no model)"
	@echo "  make test-model         Run 10 tests with real model"
	@echo "  make stats              Show TurboVec store stats"
	@echo "  make interactive        Interactive observer session"
	@echo "  make phone-guide        Show phone runtime instructions"
	@echo "  make clean              Remove venv + caches"
	@echo "  make clean-all          Remove everything (models + data too)"