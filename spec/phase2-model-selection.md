# Phase 2 — Model Selection & Fine-Tuning Plan

> **Status:** Ready for execution  
> **Date:** 2026-08-01  
> **Author:** inneri76 / Sovereign Edge AI

---

## Executive Summary

**Goal:** Fine-tune 0.8B–2.5B models specialized on Observer Core tasks (residual detection, coherence scoring, contradiction detection, ψ₀ state maintenance, structured JSON output).

**Strategy:** Start with the strongest pre-trained small model, fine-tune with QLoRA on Unsloth, quantize to GGUF IQ1_S / IQ2_XXS for phone deployment.

---

## 1. Base Model Candidates (Updated: Unsloth Catalog, July 2026)

### Tier 1 — Newest Generation (Primary Targets)

| Rank | Model Key | Params | GGUF IQ2 Size | License | Why |
|------|-----------|--------|--------------|---------|-----|
| 🥇 | **qwen3.5-2b** | 2.0B | ~450 MB | Apache 2.0 | NEWEST Qwen3.5. Best quality/size ratio |
| 🥈 | **qwen3.5-0.8b** | 0.8B | ~180 MB | Apache 2.0 | NEWEST 0.8B. Ideal lightweight Observer Core |
| 🥉 | **gemma4-e2b** | 2.0B | ~450 MB | Gemma | Google's latest 2B. Strong multilingual |

### Tier 2 — Latest Alternatives

| Rank | Model Key | Params | GGUF IQ2 Size | License | Why |
|------|-----------|--------|--------------|---------|-----|
| 4 | **ministral3-3b** | 3.0B | ~650 MB | Apache 2.0 | Mistral's newest 3B. Excellent instruct |
| 5 | **deepseek-r1-1.5b** | 1.5B | ~350 MB | MIT | R1 reasoning-distilled. Best for contradiction detection |
| 6 | **qwen3.5-4b** | 4.0B | ~900 MB | Apache 2.0 | Higher quality, mid-range+ phones |

### Tier 3 — Previous Gen (Fallbacks)

| Rank | Model Key | Params | GGUF IQ2 Size | License | Why |
|------|-----------|--------|--------------|---------|-----|
| 7 | **qwen3-1.7b** | 1.7B | ~380 MB | Apache 2.0 | Well-tested, proven |
| 8 | **smollm2-1.7b** | 1.7B | ~380 MB | Apache 2.0 | HuggingFace compact |
| 9 | **llama3.2-1b** | 1.2B | ~280 MB | Llama Community | Widest ecosystem support |
| 10 | **bonsai-8b** | 8.2B | 1150 MB | Open | Existing 1-bit native (inference only) |

### Decision: qwen3.5-2b as primary, qwen3.5-0.8b as lightweight

**Reasons:**
1. Qwen3.5 is the newest generation (July 2026), outperforming all previous gen at equal size
2. Apache 2.0 license — no restrictions
3. Pre-quantized 4-bit available on Unsloth (`unsloth/Qwen3.5-2B-Instruct-bnb-4bit`)
4. 2B IQ2_XS: ~450 MB — comfortably under 800 MB target
5. 0.8B IQ2_XXS: ~180 MB — fits any phone
6. All Unsloth models fine-tune 2x faster with 70% less VRAM

---
| 4 | Llama-3.2-1B-Instruct | 1.24B | IFEval: 57.0 | Llama 3.2 Community | Strong instruction following |
| 5 | SmolLM3-3B (fallback) | 3B | Strong multilingual | Apache 2.0 | If larger models needed |
| 6 | Bonsai-8B (existing) | 8.2B | 1-bit native, 1.15 GB | Open weights | Already in our stack |

### Decision: Qwen2.5-1.5B-Instruct as primary

**Reasons:**
1. Apache 2.0 license — no restrictions
2. MMLU-Pro 20.0 at 1.5B — punches above weight
3. IFEval 44.8 — strong instruction-following (critical for structured JSON output)
4. Widely supported by Unsloth, llama.cpp, GGUF
5. 1.54B params → QLoRA fine-tune fits on single RTX 3080 16GB
6. Quantized to IQ2_XXS: ~500 MB — under 800 MB target

---

## 2. Quantization Targets

| Quant | Format | Estimated Size (1.5B) | Quality Loss | Phone Fit |
|-------|--------|----------------------|--------------|-----------|
| IQ2_XXS | GGUF | ~400 MB | 2-5% vs FP16 | ✅ Any phone |
| IQ2_XS | GGUF | ~450 MB | 1-3% vs FP16 | ✅ Any phone |
| IQ1_S | GGUF | ~250 MB | 3-8% vs FP16 | ✅ Any phone |
| Q4_K_M | GGUF | ~900 MB | <0.5% vs FP16 | ✅ Mid-range+ |
| Native 1.58-bit | BitNet | ~350 MB | Competitive | ✅ All phones |

**Recommendation:** Train in FP16 → QLoRA → export GGUF at IQ2_XS (primary) + IQ1_S (lightweight) + Q4_K_M (quality).

---

## 3. Training Configuration

### 3.1 Environment

```
GPU: RTX 3080 Laptop 16GB VRAM
Framework: Unsloth (2x faster, free Colab T4 as fallback)
Method: QLoRA (4-bit base, LoRA adapters)
```

### 3.2 QLoRA Hyperparameters

```yaml
lora_r: 16               # LoRA rank
lora_alpha: 32           # Scaling factor
lora_dropout: 0.05       # Regularization
lora_target_modules:      # Qwen2.5 attention layers
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  effective_batch_size: 16
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  num_epochs: 3
  max_seq_length: 2048
  bf16: true              # If GPU supports it
  gradient_checkpointing: true
  
dataset:
  train_split: residuals_train.jsonl (4,019 examples)
  val_split: residuals_val.jsonl (499 examples)
  formatting: chat template → Observer Core prompt
```

### 3.3 Prompt Format (Chat Template)

```
<|im_start|>system
You are the Sovereign Edge Observer Core. Your ONLY functions are:
1. Detect residuals (gap between intent and outcome)
2. Score coherence (0.0-1.0) against six axioms
3. Detect contradictions with prior state
4. Propose minimal corrections (append-only)
5. Update invariant observer state (ψ₀)
6. Emit structured JSON output

Six Axioms (weighted):
- Awareness Is Law (20%)
- Truth Over Comfort (20%)
- Coherence Over Features (15%)
- Append-Only Memory (15%)
- Human Final Authority (15%)
- Local Sovereignty (15%)

Output ONLY valid JSON matching the Observer Core residual schema.
<|im_end|>
<|im_start|>user
Intent: {intent}
Predicted: {predicted}
Executed: {executed}
Actual: {actual}
<|im_end|>
<|im_start|>assistant
{output JSON}
<|im_end|>
```

### 3.4 Training Data Mix

| Data Type | Examples | Weight in Mix |
|-----------|----------|---------------|
| Residual detection (standard) | 4,019 | 50% |
| Coherence judgments | 2,000 | 25% |
| Contradiction pairs | 1,000 | 12% |
| Structured JSON records | 1,000 | 13% |

Observer sequences and preference pairs saved for Phase 2b (RLHF/DPO).

---

## 4. Evaluation Metrics

### 4.1 Residual-Specific Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| JSON schema compliance | ≥ 95% | % of outputs matching exact schema |
| Hard gate detection | ≥ 90% | Recall on truth violations |
| Coherence score correlation | Pearson r ≥ 0.80 | vs human judgments |
| Contradiction recall | ≥ 85% | Detecting planted contradictions |
| Correction quality | Human eval ≥ 4/5 | Minimal, append-only, specific |
| ψ₀ state consistency | 100% | No state corruption across 50+ turns |

### 4.2 Performance Metrics

| Metric | Target |
|--------|--------|
| Tokens/sec (phone, IQ2_XS) | ≥ 8 tok/s |
| RAM at inference | ≤ 2 GB |
| Disk footprint (IQ2_XS) | ≤ 500 MB |
| Disk footprint (IQ1_S) | ≤ 300 MB |

---

## 5. Execution Steps

### Step 1: Environment Setup
```bash
pip install unsloth transformers datasets accelerate peft bitsandbytes
pip install llama-cpp-python  # For GGUF export
```

### Step 2: Load & Format Data
- Load residuals_train.jsonl, coherence_judgments_train.jsonl
- Format into chat template
- 80/10/10 split already done

### Step 3: QLoRA Fine-Tune (Unsloth)
- Train on RTX 3080 or Colab T4 (free tier works)
- ~2-4 hours for 3 epochs on 4k examples
- Save LoRA adapter weights

### Step 4: Merge & Export
- Merge LoRA into base model
- Convert to GGUF: IQ2_XS, IQ1_S, Q4_K_M
- Test with llama.cpp

### Step 5: Evaluate
- Run on test split (502 examples)
- Compute residual-specific metrics
- Test on real phone hardware

### Step 6: Iterate
- If hard gate recall < 90% → add more hard-gate examples
- If JSON compliance < 95% → add more structured examples
- If coherence correlation < 0.80 → improve scoring annotations

---

## 6. Fallback Plan

If Qwen2.5-1.5B underperforms:
1. **Try Llama-3.2-1B-Instruct** — stronger IFEval (57.0) but weaker MMLU
2. **Try BitNet-b1.58-2B** — native ternary, phone-optimized, needs bitnet.cpp runtime
3. **Scale up to Qwen2.5-3B-Instruct** — 3B, MMLU-Pro 25.1, still fits phone with IQ2_XXS (~800 MB)
4. **Scale down to Qwen2.5-0.5B** — ~250 MB, as fallback for weakest phones

---

## 7. Phone Deployment

### llama.cpp Runtime

```bash
# On Android (Termux)
./llama-cli -m observer-core-1.5b-IQ2_XS.gguf \
    -p "<|im_start|>system\nYou are the Sovereign Edge Observer Core..." \
    -n 512 --temp 0.1 -t 4

# On iPhone (LLM Farm / PocketPal)
# Import GGUF → select model → set system prompt
```

### Minimum Phone Specs

| Model | RAM Needed | Storage | tok/s (est.) |
|-------|-----------|---------|-------------|
| Qwen2.5-0.5B IQ1_S | 1 GB | 250 MB | 15-25 |
| Qwen2.5-1.5B IQ2_XS | 2 GB | 450 MB | 8-15 |
| Qwen2.5-1.5B Q4_K_M | 3 GB | 900 MB | 6-12 |

---

## 8. Next: Phase 3

After Phase 2 model is trained and validated:
1. Preference optimization (DPO) on preference_pairs data
2. Multi-turn training on observer_sequences
3. Auto-evaluation pipeline
4. Model card + HuggingFace upload

---

**Phase 2 status: PLAN COMPLETE — Ready for training execution.**