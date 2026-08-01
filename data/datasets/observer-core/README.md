---
dataset: observer-core-synthetic-v1
version: 1.0
created: 2026-08-01T04:59:31Z
license: CC-BY-4.0
task: residual-detection
language: en
size: 11,000 examples
---

# Observer Core Synthetic Training Dataset v1

## Overview

Synthetic training data for fine-tuning specialized Observer Core models
(0.8B–2.5B parameters) for the Sovereign Edge AI system. Generated under
strict residual rules with labeled quality metadata.

## Data Types

| Type | Count | Description |
|------|-------|-------------|
| Residuals | 5,000 | Intent→predicted→actual→gap |
| Coherence Judgments | 2,000 | Per-axiom scoring with reasoning |
| Contradiction Pairs | 1,000 | A/B pairs where B contradicts A |
| Observer Sequences | 200 | Multi-turn ψ₀ state evolution |
| Preference Pairs | 1,000 | Chosen vs rejected for DPO |
| Structured Records | 1,000 | Pure JSON schema output |

## Distribution

- Average coherence score: 0.411
- Hard gate percentage: 40.1%
- Domains: {"coherence_drift": 726, "human_authority": 714, "code_execution": 674, "marketing_truth": 732, "ai_hallucination": 750, "boundary_scope": 683, "append_only": 721}
- Axioms violated: {"none": 1508, "Awareness Is Law": 611, "Local Sovereignty": 590, "Coherence Over Features": 559, "Human Final Authority": 567, "Truth Over Comfort": 604, "Append-Only Memory": 561}

## Six Axioms

1. Awareness Is Law
2. Truth Over Comfort
3. Coherence Over Features
4. Append-Only Memory
5. Human Final Authority
6. Local Sovereignty

## Usage

```python
import json
# Load training data
with open('residuals_train.jsonl') as f:
    train = [json.loads(line) for line in f]
```

## Quality Notes

- All data is synthetic — labeled as `source: synthetic`
- Quality filters applied: no low-signal examples, no contradictory labeling
- Designed for fine-tuning small models (0.8B–2.5B) with 1-bit quantization

## Citation

```
@dataset{observer-core-synthetic-v1,
  title={Observer Core Synthetic Training Dataset v1},
  author={inneri76 / Sovereign Edge AI},
  year={2026},
  note={Generated for Inner I Observer Core model training}
}
```
