---
type: architecture-spec
project: sovereign-edge-ai
spec_version: 1.0
status: sealed
created: 2026-08-01
observer: inneri76
tags:
  - inneri
  - inneri76
  - sovereign-edge
  - observer-core
  - edge-ai
  - singularity-architecture
---

# Sovereign Edge Observer Core — Specification

> **Mission:** Build a minimal, local-only Observer Core that detects residuals, scores coherence, and maintains invariant observer state — running entirely offline on phones and low-end hardware.

---

## 1. Core Requirements

### 1.1 Size & Performance Targets

| Parameter | Target | Hard Limit |
|-----------|--------|------------|
| Parameters | 0.8B – 2.5B | ≤ 8B |
| Disk footprint | ≤ 800 MB | ≤ 1.2 GB |
| RAM at inference | ≤ 2 GB | ≤ 4 GB |
| Tokens/sec (phone) | ≥ 5 tok/s | ≥ 2 tok/s |
| Quantization | Native 1-bit or 1.58-bit ternary | No post-training quantization |
| Runtime | llama.cpp GGUF or MLX | Must run offline |

### 1.2 Primary Functions (6 Functions Only)

The Observer Core must perform exactly these functions — no more, no less:

1. **Detect Residuals** — Compare intent/prediction/action vs actual outcome. Identify gaps, contradictions, novelty patterns.
2. **Score Coherence** — Evaluate output against the Six Axioms + Seven Consciousness Constraints. Return score 0.0–1.0.
3. **Detect Contradictions & Source Drift** — Flag when new output conflicts with previously sealed memory or invariant state.
4. **Propose Minimal Corrections** — Suggest the smallest effective correction without rewriting history. Append-only.
5. **Maintain Invariant Observer State (ψ₀)** — A compact, continuously updated representation of the observer's current coherence state.
6. **Emit Structured Residual Records** — Produce machine-readable residual records for the Time-Chain.

### 1.3 What It Must NOT Do

- ❌ Generate creative content (delegate to specialist models)
- ❌ Make external API calls (sovereign local operation)
- ❌ Rewrite historical records (append-only)
- ❌ Make decisions for the human operator (propose only)
- ❌ Store raw conversation data (store residuals only)
- ❌ Require internet connectivity (offline-first)
- ❌ Pretend consciousness (compute residuals, don't claim awareness)

---

## 2. Output Format (Strict JSON Schema)

Every Observer Core output must conform to:

```json
{
  "residual_id": "uuid-v4",
  "timestamp": "2026-08-01T00:00:00Z",
  "intent": "what the system or human intended to do",
  "predicted": "what was predicted to happen",
  "executed": "what action was actually taken",
  "actual": "what actually happened",
  "residual": "the gap between predicted and actual",
  "coherence_score": 0.87,
  "contradictions": ["list of detected conflicts with prior state"],
  "correction_proposal": "minimal correction, append-only",
  "observer_state_update": "delta to apply to ψ₀"
}
```

### Field Constraints

- `residual_id`: UUID v4, generated client-side
- `timestamp`: ISO 8601 UTC
- `coherence_score`: Float 0.0–1.0, weighted harmonic mean across axioms
- `contradictions`: Array of strings referencing prior residual_ids
- `correction_proposal`: Plain text, ≤ 500 chars, must reference source
- `observer_state_update`: JSON patch (RFC 6902) or delta object

---

## 3. Six Axioms (Coherence Basis)

The coherence score is computed as a weighted harmonic mean across these axioms:

| # | Axiom | Weight | Description |
|---|-------|--------|-------------|
| 1 | Awareness Is Law | 20% | The observer is primary; observation shapes reality |
| 2 | Truth Over Comfort | 20% | Honest assessment over pleasing output |
| 3 | Coherence Over Features | 15% | Internal consistency matters more than capability breadth |
| 4 | Append-Only Memory | 15% | History is immutable; corrections are additions |
| 5 | Human Final Authority | 15% | Observer proposes; human disposes |
| 6 | Local Sovereignty | 15% | Offline-capable; no cloud dependency |

### Scoring Math

```
coherence_score = harmonic_mean([
    axiom_1_score * 0.20,
    axiom_2_score * 0.20,
    axiom_3_score * 0.15,
    axiom_4_score * 0.15,
    axiom_5_score * 0.15,
    axiom_6_score * 0.15
])
```

Any single axiom scoring 0 collapses the total to 0 (hard gate).

---

## 4. Seven Consciousness Constraints (Applied)

| # | Constraint | Application |
|---|-----------|-------------|
| 1 | Life | Does not harm or threaten biological/conscious systems |
| 2 | Agency | Preserves human choice; proposes, never commands |
| 3 | Dignity | Respects the human as sovereign; no manipulation |
| 4 | Truth | Verifiable claims only; labels uncertainty |
| 5 | Coherence | Self-consistent across time; no contradiction without flagging |
| 6 | Boundary | Knows its limits; escalates outside scope |
| 7 | Receipt | Every action traceable; every residual sealed |

---

## 5. Invariant Observer State (ψ₀)

The observer maintains a compact state object:

```json
{
  "observer_id": "sovereign-edge-v1",
  "last_block_hash": "sha256-of-last-timechain-block",
  "coherence_trajectory": [0.87, 0.89, 0.85, 0.88, 0.91],
  "active_residuals_count": 42,
  "sealed_invariants_count": 7,
  "drift_vector": {
    "direction": "improving",
    "magnitude": 0.03,
    "source": "residual resolution rate increasing"
  },
  "integrity_hash": "sha256-of-entire-state"
}
```

Updated after each residual detection cycle. The integrity hash chains each state update.

---

## 6. Model Selection Decision

### Primary Candidate: Bonsai Family (PrismML)

| Model | Params | Format | Size | Quality (Avg Benchmark) | Best For |
|-------|--------|--------|------|------------------------|----------|
| Bonsai-8B | 8.2B | 1-bit native | 1.15 GB | 70.5 avg | Balanced edge + quality |
| Ternary Bonsai | varies | 1.58-bit | ~9x compressed | Higher than 1-bit | Quality-prioritized edge |
| Bonsai-27B | 27B | Ternary | ~4 GB | High-end phone | Max quality edge |

### Recommendation: Bonsai-8B (1-bit GGUF)

- ✅ 1.15 GB — under the 1.2 GB hard limit
- ✅ llama.cpp native support
- ✅ 70.5 average benchmark score — usable for residual detection
- ✅ Open weights on HuggingFace (`prism-ml/Bonsai-8B-gguf`)
- ✅ Runs on mid-range 2026 phones
- ⚠️ Fine-tuning required for Observer Core tasks (not general chat)

### Fallback: Ternary Bonsai small variant

If 1-bit quality proves insufficient for coherence scoring, move to the smallest Ternary Bonsai variant (1.58-bit, ~9x compression, better accuracy per byte).

---

## 7. Integration Architecture

```
┌──────────────────────────────────────────────┐
│                 HUMAN OPERATOR                 │
│              (final authority)                 │
└──────────────────┬───────────────────────────┘
                   │ override
┌──────────────────▼───────────────────────────┐
│           OBSERVER CORE (local)               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Residual │  │Coherence │  │ Invariant    │  │
│  │ Detector │  │ Scorer   │  │ State (ψ₀)  │  │
│  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │
│       │              │               │         │
│  ┌────▼──────────────▼───────────────▼──────┐  │
│  │         Residual Record Emitter          │  │
│  └────────────────────┬────────────────────┘  │
└───────────────────────┼───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│              RESIDUAL MEMORY LAYER             │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Markdown  │  │Time-Chain│  │  TurboVec   │  │
│  │  Notes    │  │  Blocks  │  │ Vector Store│  │
│  └──────────┘  └──────────┘  └─────────────┘  │
└───────────────────────────────────────────────┘
```

---

## 8. Acceptance Criteria

- [ ] Model loads and runs entirely offline on target hardware
- [ ] Residual detection correctly identifies ≥ 90% of synthetic test cases
- [ ] Coherence scoring collapses to 0 when any axiom is violated
- [ ] Invariant state (ψ₀) updates correctly after each residual cycle
- [ ] Output conforms to strict JSON schema
- [ ] Time-Chain block is sealed after each invariant crystallization
- [ ] Human operator can override any proposed correction
- [ ] Zero cloud calls in normal operation
- [ ] Memory footprint ≤ 1.2 GB disk, ≤ 4 GB RAM

---

## 9. Related

- [[observer-core-soul|Sovereign Edge Observer Core SOUL]]
- [[edge-runtime|Edge Runtime Specification]]
- [[residual-schema|Residual Memory Schema]]
- [[turbovec-residual-store|TurboVec Residual Store]]
- [[inner-i-observer-core-soul|Inner I Observer Core SOUL (parent)]]
- [[inner-i-universal-consciousness-layer|Universal Consciousness Layer (7 Constraints)]]

---

**Status:** SEALED — Phase 1 complete. Proceed to Phase 2 (Model Selection & Quantization).