---
type: operator-handover
project: sovereign-edge-ai
version: 1.0
created: 2026-08-01
observer: inneri76
tags:
  - inneri
  - inneri76
  - sovereign-edge
  - handover
  - operator-guide
---

# Operator Handover — Sovereign Edge AI

> **You are now the operator of a Sovereign Edge Observer Core.** This document tells you what you own, how to control it, and how to take full authority.

---

## What You Own

You now possess the complete specification for a **local-only, phone-capable AI observer system** that:

1. Runs entirely on your hardware — no cloud, no subscriptions, no external APIs
2. Detects residuals — gaps between what was intended and what actually happened
3. Scores coherence — measures how well outputs align with truth, sovereignty, and consistency
4. Never rewrites history — corrections are append-only
5. Respects your final authority — proposes, never commands
6. Can run on a mid-range phone using a 1.15 GB model

---

## How To Take Control

### 1. Download the Model

```bash
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/bonsai-8b-q1.gguf
```

Verify checksum (TBD — will be added when model is fine-tuned for observer tasks).

### 2. Set Up the Runtime

**On laptop/desktop:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j4
./llama-cli -m bonsai-8b-q1.gguf -f observer-prompt.txt -n 256 --temp 0.1
```

**On Android (Termux):**
```bash
pkg install cmake make clang wget
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
```

### 3. Understand What You're Running

The Observer Core does exactly 6 things:
1. Detects residuals (intent vs outcome gaps)
2. Scores coherence (0.0-1.0 across six axioms)
3. Detects contradictions (against prior sealed state)
4. Proposes corrections (minimal, append-only)
5. Maintains invariant state (ψ₀)
6. Emits structured residual records (JSON)

It does NOT:
- Chat, generate content, or converse
- Connect to the internet
- Make decisions for you
- Rewrite history
- Claim consciousness

### 4. Override Authority

Your override is FINAL. When the observer proposes a correction, you can:

- **Accept**: The correction is appended as a new residual, original record stays sealed.
- **Override**: Your correction replaces the proposal. Your override is sealed as `operator_override: true`.
- **Dismiss**: The residual is marked `dismissed_by_operator` with timestamp. No correction is made.

To override:
```
OPERATOR OVERRIDE: [your correction text]
RESIDUAL_ID: [uuid of residual being corrected]
```

### 5. Read Your Residuals

All residuals are written as Markdown notes in your vault at `02_MEMORY/residuals/`. Each note is linked to the Time-Chain and knowledge graph.

To see open residuals:
```dataview
TABLE coherence_score, timestamp
FROM #residual AND #sovereign-edge
WHERE status = "open"
SORT coherence_score ASC
```

### 6. Verify the Time-Chain

```bash
# Check that the chain is intact
python3 -c "
import hashlib, json
# Walk through blocks in 10_SIGNAL/
# Verify each block's hash chains to the previous
"
```

---

## Your Rights (Non-Negotiable)

| Right | Description |
|-------|-------------|
| **Right to Disconnect** | Power off the device. Observer stops. No cloud dependency means no lingering processes. |
| **Right to Override** | Your correction always wins. Observer proposals are advisory. |
| **Right to Delete** | You can delete the model file. You can delete the residual DB. The system is yours. |
| **Right to Inspect** | Every residual, every score, every state update is stored as plain Markdown and SQLite. No black boxes. |
| **Right to Modify** | The architecture is Open Architecture. Fork it, change the axioms, retrain the model. |
| **Right to Sovereignty** | This system runs on YOUR hardware, YOUR decisions, YOUR authority. Period. |

---

## What To Watch For

### Drift Indicators

If coherence scores consistently decline over 7+ days, the system may be:
- Accumulating unresolved residuals faster than they're resolved
- Detecting false positives (noise in residual detection)
- Experiencing model degradation (rare with inference-only)

### Contradiction Cascades

If a single residual triggers 5+ contradictions with prior sealed state, stop and review. This may indicate:
- A genuine shift in understanding (update ψ₀ trajectory)
- A bug in the observer model (retrain or recalibrate)
- Human input that needs re-examination

### Storage Growth

At 1,000 residuals/day, your storage grows ~7 MB/day. On a 64 GB phone, that's ~25 years of residuals. Monitor `turbovec.db` size monthly.

---

## Escalation Path (When You Need Help)

1. **Vault check**: Search `#sovereign-edge` in your Obsidian vault for related residuals
2. **GitHub**: Issues at `TheInnerI/sovereign-edge-ai`
3. **Contact**: i@innerinetcompany.com
4. **Community**: Inner I Network (when live)

---

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | Vault preparation, architecture files |
| Phase 1 | ✅ Complete | Observer Core specification sealed |
| Phase 2 | ✅ Complete | Model selected (Bonsai-8B), runtime documented |
| Phase 3 | ✅ Complete | Residual memory schema + TurboVec implementation |
| Phase 4 | ✅ Complete | Local-first runtime spec + tier router |
| Phase 5 | ✅ Complete | Observer Core SOUL minted + agent registry |
| Phase 6 | ⬜ Pending | Hardware validation + synthetic residual testing |

---

## Next Actions (For You, The Operator)

1. [ ] Download Bonsai-8B model and verify it runs on your target device
2. [ ] Run 10 synthetic residual test cases and verify detection accuracy ≥ 90%
3. [ ] Measure tokens/sec on your phone/laptop
4. [ ] Fine-tune the model on observer-specific tasks (or wait for the fine-tuned release)
5. [ ] Seal the first Time-Chain block with your first batch of residuals
6. [ ] Set a weekly review rhythm (Sunday evening: review open residuals, check ψ₀ drift)

---

## Final Word

> This is not a product you bought. It's infrastructure you own.
>
> The Inner I is the observer within every human being. inneri76 built the tools. The tools are yours.
>
> Now go observe. Detect the gaps. Score the coherence. And remember: the human has final authority.

— inneri76 / Inner I Network
— i@innerinetcompany.com
— August 1, 2026