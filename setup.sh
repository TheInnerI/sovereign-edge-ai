#!/bin/bash
# Sovereign Edge AI — One-Command Setup
# Installs everything needed to run the observer locally.
set -e

echo "🛡️  Sovereign Edge AI — Setup"
echo "================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 1. Create venv
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# 2. Install deps
echo "📥 Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

# 3. Train embedder
echo "🧠 Training TF-IDF embedder..."
.venv/bin/python run.py train-embedder

# 4. Check for model
if [ -f "models/Bonsai-8B-Q1_0.gguf" ]; then
    echo "✅ Model found: models/Bonsai-8B-Q1_0.gguf"
else
    echo ""
    echo "📥 Model not found. Download Bonsai-8B (1.15 GB)?"
    echo "   Run: make download-model"
    echo ""
fi

# 5. Try Ollama import
if [ -f "models/Bonsai-8B-Q1_0.gguf" ] && command -v ollama &> /dev/null; then
    echo "📦 Importing into Ollama..."
    ollama create sovereign-observer -f Modelfile 2>/dev/null && \
        echo "✅ Ollama model ready: ollama run sovereign-observer" || \
        echo "⚠️  Ollama import skipped (may not support Q1_0 format yet)"
fi

# 6. Run tests
echo ""
echo "🧪 Running synthetic tests..."
.venv/bin/python tests/test_synthetic.py

echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  make test-synthetic     # Run tests (no model needed)"
echo "  make download-model     # Download Bonsai-8B (1.15 GB)"
echo "  make interactive        # Interactive observer session"
echo "  cat PHONE-GUIDE.md      # Phone runtime instructions"