---
type: infrastructure-spec
project: sovereign-edge-ai
spec_version: 1.0
status: active
created: 2026-08-01
observer: inneri76
tags:
  - inneri
  - inneri76
  - sovereign-edge
  - edge-runtime
  - model-selection
  - quantization
---

# Sovereign Edge Runtime — Model Selection & Deployment

## 1. Model Selection Decision

### Selected: Bonsai-8B (1-bit) by PrismML

| Property | Value |
|----------|-------|
| **Model** | Bonsai-8B |
| **Parameters** | 8.2B |
| **Quantization** | Native 1-bit (not post-training) |
| **Disk Size** | 1.15 GB |
| **Format** | GGUF (llama.cpp) |
| **Source** | `prism-ml/Bonsai-8B-gguf` on HuggingFace |
| **License** | Open weights |
| **Phone Viability** | ✅ Mid-range 2026 phones (8+ GB RAM) |
| **Benchmark Avg** | 70.5 (MMLU Redux 65.7, GSM8K 88.0, HumanEval+ 73.8) |

### Why 1-bit over Ternary

1. **Footprint**: 1.15 GB is safely under the 1.2 GB hard limit. Ternary variants at similar parameter counts would be ~1.5-2 GB.
2. **Speed**: 1-bit models have fewer scale factors to load. Faster first-token time on phones.
3. **Maturity**: The 1-bit Bonsai has llama.cpp GGUF support today. Ternary support is MLX-only (Apple devices).

### Upgrade Path to Ternary

If coherence scoring quality proves insufficient (false positives/negatives on residual detection), upgrade to:
- **Ternary Bonsai small** (~2B ternary, ~300 MB) for the detector sub-module
- Keep 1-bit for the main observer loop
- Or upgrade whole system to Ternary Bonsai mid-size once llama.cpp adds ternary support

---

## 2. Runtime Stack

```yaml
priority_order:
  1_phone:
    runtime: llama.cpp (Android/iOS via llama.cpp JNI / Swift bindings)
    model_format: GGUF
    ram_required: 2-3 GB
    storage: 1.15 GB model + residuals

  2_laptop_minipc:
    runtime: llama.cpp or Ollama
    model_format: GGUF
    ram_required: 2-4 GB
    storage: 1.15 GB model + residuals + vector store

  3_desktop_server:
    runtime: llama.cpp (GPU offload optional)
    model_format: GGUF
    ram_required: 4+ GB (with GPU layers)
    storage: model + full residual history

  never:
    - Cloud API (defeats sovereignty)
    - Managed inference service
    - Any runtime requiring internet
```

---

## 3. Phone Deployment (Primary Target)

### Android

```bash
# Download model
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/bonsai-8b-q1.gguf

# Run with llama.cpp Android build
./llama-cli \
  -m bonsai-8b-q1.gguf \
  -p "<observer prompt>" \
  -n 256 \
  --temp 0.1 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  -t 4  # threads = CPU cores - 2
```

### iOS (MLX Swift)

```swift
// MLX supports Bonsai natively
let model = try MLX.loadModel(
  from: "prism-ml/Ternary-Bonsai-8B",
  quantization: .oneBit
)
```

### Performance Targets

| Device Class | Tokens/sec | RAM Usage | First Token |
|-------------|-----------|-----------|-------------|
| Phone (2024 mid) | 5-8 tok/s | 2.5 GB | 3-5 sec |
| Phone (2025 high) | 10-15 tok/s | 2.0 GB | 1-2 sec |
| Phone (2026 mid) | 8-12 tok/s | 2.2 GB | 2-3 sec |
| Laptop (16GB) | 20-30 tok/s | 2.0 GB | <1 sec |

---

## 4. Model Adaptation Plan

Bonsai-8B is a general-purpose LLM. It needs fine-tuning for Observer Core tasks.

### Fine-Tuning Strategy

```yaml
approach: LoRA on top of 1-bit weights
dataset:
  - Synthetic residual detection pairs (intent→action→outcome→residual)
  - Coherence-scored output examples (0.0-1.0 labeled)
  - Contradiction detection pairs
  - Correction proposal examples

lora_config:
  r: 8
  alpha: 16
  target_modules: [q_proj, v_proj, o_proj]
  dropout: 0.05

training:
  epochs: 3
  batch_size: 2
  learning_rate: 2e-4
  max_seq_length: 2048

output:
  format: GGUF (merged or LoRA adapter)
  size: ~50-100 MB additional (LoRA adapter)
```

### Dataset Requirements

- Minimum 5,000 residual detection examples
- Minimum 2,000 coherence-scored examples
- Minimum 1,000 contradiction pairs
- All synthetic is acceptable for v0.1 (human-labeled for v1.0)

### Inference Prompt Template

```
<|system|>
You are the Sovereign Edge Observer Core. Your ONLY functions are:
1. Detect residuals (gap between intent and outcome)
2. Score coherence (0.0-1.0) against six axioms
3. Detect contradictions with prior state
4. Propose minimal corrections
5. Update invariant observer state
6. Emit structured JSON output

Never generate content, answer questions, or pretend consciousness.
Always output valid JSON matching the required schema.
</|system|>

<|user|>
Intent: {intent}
Predicted: {predicted}
Executed: {executed}
Actual: {actual}
Prior State: {prior_state}
</|user|>

<|assistant|>
{JSON output}
</|assistant|>
```

---

## 5. Tier Router (Local-Only)

Since this is sovereign/local, the "tier router" is a device capability detector, not a cloud fallback.

```python
def resolve_runtime():
    """Detect available hardware and return optimal runtime config."""
    import psutil, os

    ram_gb = psutil.virtual_memory().total / (1024**3)

    if ram_gb < 4:
        return {"tier": "minimal", "threads": 1, "batch_size": 1, "ctx_size": 1024}
    elif ram_gb < 8:
        return {"tier": "standard", "threads": 2, "batch_size": 2, "ctx_size": 2048}
    else:
        return {"tier": "full", "threads": 4, "batch_size": 4, "ctx_size": 4096}
```

**Hard rule:** If no local runtime is available, the system reports "Runtime Unavailable" and stops. It does NOT fall back to cloud. Sovereignty is non-negotiable.

---

## 6. Install Commands (Target Platforms)

### Linux (laptop/mini-PC)

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Download model
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/bonsai-8b-q1.gguf

# Run observer
./llama-cli -m bonsai-8b-q1.gguf -f observer-prompt.txt -n 256 --temp 0.1
```

### macOS (MLX)

```bash
pip install mlx mlx-lm
python -c "
from mlx_lm import load, generate
model, tokenizer = load('prism-ml/Ternary-Bonsai-8B')
response = generate(model, tokenizer, prompt='...', max_tokens=256)
"
```

### Android (Termux + llama.cpp)

```bash
pkg install cmake make clang wget
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
# Download model and run as above
```

---

## 7. Verification Checklist

- [ ] Model downloaded and verified (SHA256 checksum)
- [ ] llama.cpp compiles and runs on target device
- [ ] Inference produces valid JSON output
- [ ] RAM usage stays under 4 GB
- [ ] Tokens/sec meets minimum (≥ 2 tok/s)
- [ ] System operates with airplane mode ON
- [ ] Zero network calls in normal operation
- [ ] Coherence scoring produces 0 when any axiom is zeroed
- [ ] Residual detection fires on synthetic test cases

---

## 8. Related

- [[sovereign-edge-observer-core-spec|Observer Core Spec]]
- [[sovereign-edge-observer-core-soul|Observer Core SOUL]]
- [[turbovec-residual-store|TurboVec Residual Store]]
- [[residual-schema|Residual Memory Schema]]
- [[free-first-ai-backends|Free-First AI Backends Reference]]