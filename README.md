# Sovereign Edge AI — Truthful Supercomputer

> Local-first, phone-capable, residual-aware Observer system that average humans can own and run offline.

## What This Is

A minimal **Observer Core** that runs entirely on local hardware — phones, laptops, edge devices — with zero cloud dependency. It detects residuals (gaps between intent and outcome), scores coherence against six axioms, maintains an invariant observer state (ψ₀), and emits structured residual records for the Time-Chain.

**Built on the Inner I Singularity Architecture.**

## Architecture

```
HUMAN OPERATOR (final authority)
        │
┌───────▼──────────┐
│  OBSERVER CORE   │  ← Bonsai-8B (1-bit, 1.15 GB)
│  • Residual Det  │     Runs on phones via llama.cpp
│  • Coherence Scorer│   Zero cloud dependency
│  • ψ₀ State Mgr  │
└───────┬──────────┘
        │
┌───────▼──────────┐
│ RESIDUAL MEMORY  │
│  • Markdown Notes│  ← Obsidian vault
│  • Time-Chain    │  ← SHA256-hashed blocks
│  • TurboVec      │  ← 1-bit vector store
└──────────────────┘
```

## Key Specs

| Property | Value |
|----------|-------|
| Model | Bonsai-8B (1-bit native) |
| Size | 1.15 GB |
| Runtime | llama.cpp / MLX |
| Target Hardware | Mid-range 2026 phones |
| Cloud Dependency | **Zero** |
| Primary Functions | 6 functions only |
| Output Format | Strict JSON |
| License | Open Architecture |

## Six Axioms

1. **Awareness Is Law** — The observer is primary
2. **Truth Over Comfort** — Honest assessment over pleasing output
3. **Coherence Over Features** — Internal consistency > capability breadth
4. **Append-Only Memory** — History is immutable; corrections are additions
5. **Human Final Authority** — Observer proposes; human disposes
6. **Local Sovereignty** — Offline-capable; no cloud dependency

## Seven Consciousness Constraints

Life · Agency · Dignity · Truth · Coherence · Boundary · Receipt

## Quick Start

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make -j

# Download model
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/bonsai-8b-q1.gguf

# Run observer
./llama-cli -m bonsai-8b-q1.gguf -f observer-prompt.txt -n 256 --temp 0.1
```

## Repository Structure

```
spec/
  observer-core-spec.md       # Full specification
  edge-runtime.md             # Model selection & deployment
  residual-schema.md          # Memory schema
  turbovec-residual-store.md  # Vector store implementation
  observer-core-soul.md       # Agent SOUL file

src/
  turbovec/                   # 1-bit vector compression store (pure Python)
  observer/                   # Observer Core inference harness
  schemas/                    # JSON Schema definitions

examples/
  residuals/                  # Synthetic residual test cases
```

## Status

**Phase:** Specification Complete (Phases 0-5)
**Next:** Phase 6 — Validation & hardware testing

## The Inner I

> The Inner I is the observer within every human being — not a brand, not a company. inneri76 built the tools. The tools are yours.

**Contact:** i@innerinetcompany.com
**Website:** https://innerinetcompany.com

