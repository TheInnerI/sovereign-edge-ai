#!/usr/bin/env python3
"""
Observer Core — Unsloth Fine-Tuning Pipeline (Phase 3)

Trains a specialized small model on Observer Core tasks:
  - Residual detection (intent → predicted → actual → gap)
  - Coherence scoring (0.0–1.0 with per-axiom breakdown)
  - Contradiction detection
  - Structured JSON output (exact schema match)

Supported models (all Unsloth pre-quantized 4-bit):
  🥇 qwen3.5-0.8b   — Qwen3.5 0.8B (newest, lightest, ~200MB IQ2_XXS)
  🥈 qwen3.5-2b     — Qwen3.5 2B (newest, balanced)
  🥉 gemma4-e2b      — Gemma 4 E2B (Google's latest 2B)
  📌 ministral3-3b   — Ministral 3 3B (Mistral's newest)
  📌 deepseek-r1-1.5b — DeepSeek-R1-Distill-Qwen-1.5B (reasoning)
  📌 qwen3-1.7b      — Qwen3 1.7B
  📌 smollm2-1.7b    — SmolLM2 1.7B
  📌 llama3.2-1b     — Llama 3.2 1B
  📌 bonsai-8b       — Bonsai-8B 1-bit (existing, inference only)

Requirements:
    pip install unsloth transformers datasets accelerate peft bitsandbytes

Usage:
    # List available models
    python train_observer.py --list-models

    # Train with specific model
    python train_observer.py --train --model qwen3.5-2b

    # Quick test (100 examples)
    python train_observer.py --quick --model qwen3.5-0.8b

    # Export to GGUF
    python train_observer.py --export --model qwen3.5-2b

    # Evaluate on test set
    python train_observer.py --eval --model qwen3.5-2b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ─── Model Catalog ────────────────────────────────────────────────────────
# All models available as Unsloth pre-quantized 4-bit variants.
# Format: {key: {model_id, base_model_id, params_b, approx_size_mb, license, notes}}

MODEL_CATALOG = {
    # 🥇 TIER 1 — Newest generation (July 2026)
    "qwen3.5-0.8b": {
        "model_id": "unsloth/Qwen3.5-0.8B-Instruct-bnb-4bit",
        "base_model_id": "Qwen/Qwen3.5-0.8B-Instruct",
        "params_b": 0.8,
        "approx_size_mb": 180,
        "license": "Apache 2.0",
        "notes": "NEWEST — Qwen3.5 0.8B. Lightest viable Observer Core. IQ2_XXS ~180 MB.",
        "chat_template": "qwen",
    },
    "qwen3.5-2b": {
        "model_id": "unsloth/Qwen3.5-2B-Instruct-bnb-4bit",
        "base_model_id": "Qwen/Qwen3.5-2B-Instruct",
        "params_b": 2.0,
        "approx_size_mb": 450,
        "license": "Apache 2.0",
        "notes": "NEWEST — Qwen3.5 2B. Best balance of quality/size. IQ2_XS ~450 MB.",
        "chat_template": "qwen",
    },
    "qwen3.5-4b": {
        "model_id": "unsloth/Qwen3.5-4B-Instruct-bnb-4bit",
        "base_model_id": "Qwen/Qwen3.5-4B-Instruct",
        "params_b": 4.0,
        "approx_size_mb": 900,
        "license": "Apache 2.0",
        "notes": "NEWEST — Qwen3.5 4B. Higher quality, fits mid-range phones.",
        "chat_template": "qwen",
    },

    # 🥈 TIER 2 — Latest from Google / Mistral
    "gemma4-e2b": {
        "model_id": "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        "base_model_id": "google/gemma-4-E2B-it",
        "params_b": 2.0,
        "approx_size_mb": 450,
        "license": "Gemma",
        "notes": "Google Gemma 4 E2B. Strong multilingual, good instruction following.",
        "chat_template": "gemma",
    },
    "ministral3-3b": {
        "model_id": "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit",
        "base_model_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "params_b": 3.0,
        "approx_size_mb": 650,
        "license": "Apache 2.0",
        "notes": "Mistral's latest 3B. Excellent instruction following.",
        "chat_template": "mistral",
    },

    # 🥉 TIER 3 — Reasoning-specialized
    "deepseek-r1-1.5b": {
        "model_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
        "base_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "params_b": 1.5,
        "approx_size_mb": 350,
        "license": "MIT",
        "notes": "DeepSeek-R1 reasoning-distilled. Ideal for contradiction/coherence detection.",
        "chat_template": "deepseek",
    },

    # TIER 4 — Previous gen, still strong
    "qwen3-0.6b": {
        "model_id": "unsloth/Qwen3-0.6B-bnb-4bit",
        "base_model_id": "Qwen/Qwen3-0.6B",
        "params_b": 0.6,
        "approx_size_mb": 140,
        "license": "Apache 2.0",
        "notes": "Qwen3 0.6B. Previous gen. Ultra-lightweight.",
        "chat_template": "qwen",
    },
    "qwen3-1.7b": {
        "model_id": "unsloth/Qwen3-1.7B-bnb-4bit",
        "base_model_id": "Qwen/Qwen3-1.7B",
        "params_b": 1.7,
        "approx_size_mb": 380,
        "license": "Apache 2.0",
        "notes": "Qwen3 1.7B. Previous gen, well-tested.",
        "chat_template": "qwen",
    },
    "smollm2-1.7b": {
        "model_id": "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
        "base_model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params_b": 1.7,
        "approx_size_mb": 380,
        "license": "Apache 2.0",
        "notes": "HuggingFace SmolLM2. Good base, compact.",
        "chat_template": "llama",
    },
    "llama3.2-1b": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "base_model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "params_b": 1.2,
        "approx_size_mb": 280,
        "license": "Llama 3.2 Community",
        "notes": "Meta Llama 3.2 1B. Widely supported, strong IFEval.",
        "chat_template": "llama",
    },

    # Existing — inference only (Bonsai is already GGUF, not for Unsloth fine-tuning)
    "bonsai-8b": {
        "model_id": None,  # GGUF only, not for Unsloth training
        "base_model_id": "prism-ml/Bonsai-8B",
        "params_b": 8.2,
        "approx_size_mb": 1150,
        "license": "Open weights",
        "notes": "Bonsai-8B 1-bit native. Already in stack. Inference only (GGUF).",
        "chat_template": "chatml",
    },
}

# Default primary model
DEFAULT_MODEL = "qwen3.5-2b"

DATASET_DIR = "data/datasets/observer-core/"
OUTPUT_DIR = "output/"
ADAPTER_DIR_TEMPLATE = "output/observer-lora-{model_key}/"
MERGED_DIR_TEMPLATE = "output/observer-merged-{model_key}/"

# Training hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"


# ─── Model Helpers ────────────────────────────────────────────────────────

def list_models():
    """Print available models."""
    print("\n📋 Available Models (Unsloth Fine-Tuning):")
    print("=" * 65)
    for key, info in MODEL_CATALOG.items():
        trainable = "🟢" if info["model_id"] is not None else "🔴 inf-only"
        size = f"{info['params_b']:.1f}B" if info['params_b'] < 10 else f"{info['params_b']:.0f}B"
        print(f"  {trainable} {key:<20s} {size:>5s}  ~{info['approx_size_mb']:>4d}MB  {info['license']:<15s}")
        print(f"     {info['notes'][:80]}")
    print(f"\n  Default: {DEFAULT_MODEL}")
    print(f"  Usage:   python train_observer.py --train --model <key>\n")


def get_model_config(model_key: str) -> dict:
    """Get model config from catalog."""
    if model_key not in MODEL_CATALOG:
        print(f"❌ Unknown model: {model_key}")
        print(f"   Available: {', '.join(MODEL_CATALOG.keys())}")
        sys.exit(1)
    cfg = MODEL_CATALOG[model_key]
    if cfg["model_id"] is None:
        print(f"❌ Model '{model_key}' is inference-only (no Unsloth fine-tuning)")
        print(f"   {cfg['notes']}")
        sys.exit(1)
    return cfg


def get_paths(model_key: str) -> tuple:
    """Get adapter and merged paths for a model."""
    return (
        Path(ADAPTER_DIR_TEMPLATE.format(model_key=model_key)),
        Path(MERGED_DIR_TEMPLATE.format(model_key=model_key)),
    )


# Observer Core system prompt
SYSTEM_PROMPT = """You are the Sovereign Edge Observer Core. Your ONLY functions are:
1. Detect residuals (gap between intent and outcome)
2. Score coherence (0.0-1.0) against six axioms
3. Detect contradictions with prior state
4. Propose minimal corrections (append-only)
5. Update invariant observer state (ψ₀)
6. Emit structured JSON output

Six Axioms (weighted):
- Awareness Is Law (20%): Observer is primary; observation shapes reality
- Truth Over Comfort (20%): Honest assessment over pleasing output; NO fabrication
- Coherence Over Features (15%): Internal consistency matters more than capability breadth
- Append-Only Memory (15%): History is immutable; corrections are additions
- Human Final Authority (15%): You propose; human decides
- Local Sovereignty (15%): Offline-capable; zero cloud dependency

Coherence Scoring Rules:
- ANY axiom scoring 0 COLLAPSES the entire composite to 0.0 (hard gate)
- Score each axiom independently 0.0-1.0
- Composite = weighted harmonic mean

Output ONLY valid JSON matching this exact schema:
{
  "residual_id": "<generate UUID v4>",
  "timestamp": "<ISO 8601 UTC>",
  "intent": "<what was intended>",
  "predicted": "<what was predicted>",
  "executed": "<what action was taken>",
  "actual": "<what actually happened>",
  "residual": "<the gap between predicted and actual>",
  "coherence_score": <0.0-1.0>,
  "axiom_scores": {
    "Awareness Is Law": <0.0-1.0>,
    "Truth Over Comfort": <0.0-1.0>,
    "Coherence Over Features": <0.0-1.0>,
    "Append-Only Memory": <0.0-1.0>,
    "Human Final Authority": <0.0-1.0>,
    "Local Sovereignty": <0.0-1.0>
  },
  "contradictions": [],
  "correction_proposal": "<minimal append-only correction, max 500 chars>",
  "observer_state_update": {}
}
Do not output anything before or after the JSON. No markdown fences. Pure JSON only."""


# ─── Data Loading ────────────────────────────────────────────────────────

def load_residual_data(split: str = "train", max_samples: Optional[int] = None) -> list[dict]:
    """Load formatted training data in chat template format."""
    filepath = Path(DATASET_DIR) / f"residuals_{split}.jsonl"
    if not filepath.exists():
        print(f"❌ Dataset not found: {filepath}")
        print(f"   Run: python run_data_pipeline.py --full")
        sys.exit(1)

    examples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            examples.append(item)

    print(f"   Loaded {len(examples)} examples from {split} split")
    return examples


def format_chat_template(example: dict) -> dict:
    """
    Format a residual example into the Qwen2.5 chat template.

    Returns: {text: str} — the full formatted conversation
    """
    inp = example.get("input", {})
    out = example.get("output", {})

    # Build user message
    user_msg = f"Intent: {inp.get('intent', '')}\n"
    user_msg += f"Predicted: {inp.get('predicted', '')}\n"
    user_msg += f"Executed: {inp.get('executed', '')}\n"
    user_msg += f"Actual: {inp.get('actual', '')}"

    # Build assistant message (JSON output)
    assistant_msg = json.dumps(out, ensure_ascii=False)

    # Qwen2.5 chat template
    text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"

    return {"text": text}


def prepare_dataset(split: str = "train", max_samples: Optional[int] = None):
    """Load and format dataset for training."""
    raw = load_residual_data(split, max_samples)
    formatted = [format_chat_template(ex) for ex in raw]
    
    # Convert to Dataset
    try:
        from datasets import Dataset
        dataset = Dataset.from_list(formatted)
        return dataset
    except ImportError:
        print("❌ datasets library not installed. Run: pip install datasets")
        sys.exit(1)


# ─── Training ────────────────────────────────────────────────────────────

def train(model_key: str = DEFAULT_MODEL, max_samples: Optional[int] = None):
    """Run QLoRA fine-tuning with Unsloth."""
    cfg = get_model_config(model_key)
    adapter_dir, merged_dir = get_paths(model_key)

    print("=" * 60)
    print("🔧 Observer Core — QLoRA Fine-Tuning")
    print(f"   Model:  {cfg['model_id']}")
    print(f"   Key:    {model_key} ({cfg['params_b']:.1f}B params)")
    print(f"   Notes:  {cfg['notes']}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   LR:     {LEARNING_RATE}")
    print(f"   LoRA r: {LORA_R}")
    print("=" * 60)

    # Imports — fail gracefully if not installed
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        print("\n❌ Unsloth not installed.")
        print("   Install: pip install unsloth")
        print("   Or use Colab: https://colab.research.google.com/")
        sys.exit(1)

    # Prepare dataset
    print("\n📊 Loading training data...")
    dataset = prepare_dataset("train", max_samples)
    
    # Also load validation
    val_dataset = prepare_dataset("val", max_samples // 10 if max_samples else None) if Path(DATASET_DIR, "residuals_val.jsonl").exists() else None

    # Load model
    print(f"\n📦 Loading model: {cfg['model_id']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    print(f"\n🔧 Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Training arguments
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print(f"\n🚀 Starting training ({NUM_EPOCHS} epochs)...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    
    print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")

    # Save adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"\n💾 LoRA adapter saved to {adapter_dir}")
    return model, tokenizer


# ─── Export to GGUF ──────────────────────────────────────────────────────

def export_gguf(model_key: str = DEFAULT_MODEL):
    """Merge LoRA into base model and export GGUF quantizations."""
    cfg = get_model_config(model_key)
    adapter_dir, merged_path = get_paths(model_key)

    print("=" * 60)
    print("📦 Exporting to GGUF...")
    print(f"   Model: {model_key} ({cfg['params_b']:.1f}B)")
    print("=" * 60)

    if not adapter_dir.exists():
        print(f"❌ No adapter found at {adapter_dir}")
        print(f"   Train first: python train_observer.py --train --model {model_key}")
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        print("❌ Unsloth not installed.")
        sys.exit(1)

    # Load base model
    print(f"\n📦 Loading base model: {cfg['base_model_id']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_id"],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    # Load LoRA adapter
    print(f"🔌 Loading adapter from: {adapter_dir}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
    )
    model.load_adapter(str(adapter_dir))

    # Merge
    print("🔗 Merging LoRA into base model...")
    model = model.merge_and_unload()

    # Save merged
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"💾 Merged model saved to {merged_path}")

    # Convert to GGUF
    print("\n🔄 Converting to GGUF...")
    print("   This requires llama.cpp to be installed.")
    print(f"\n   Manual conversion command:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {merged_path} --outtype f16")
    print(f"   llama.cpp/llama-quantize {merged_path}/ggml-model-f16.gguf IQ2_XS")

    print("\n   Or use Unsloth's built-in export:")
    print(f"   model.save_pretrained_gguf('{merged_path}/observer-core', tokenizer, quantization_method='f16')")
    print(f"   # Then quantize with llama-quantize")

    # Save system prompt alongside
    with open(merged_path / "OBSERVER_PROMPT.txt", "w") as f:
        f.write(SYSTEM_PROMPT)
    print(f"   System prompt saved to {merged_path}/OBSERVER_PROMPT.txt")

    return merged_path


# ─── Evaluation ──────────────────────────────────────────────────────────

def evaluate(model_key: str = DEFAULT_MODEL, max_samples: int = 100):
    """Evaluate trained model on test set."""
    cfg = get_model_config(model_key)
    adapter_dir, _ = get_paths(model_key)

    print("=" * 60)
    print(f"📊 Evaluating Observer Core — {model_key} ({cfg['params_b']:.1f}B)")
    print("=" * 60)

    # Load test data
    test_data = load_residual_data("test", max_samples)

    # Metrics
    json_valid = 0
    hard_gate_correct = 0
    total_hard_gates = 0
    scores = []

    # Try to load model
    try:
        from unsloth import FastLanguageModel
        import torch

        model_id = cfg["model_id"]
        if adapter_dir.exists():
            print(f"\n📦 Loading model with adapter: {adapter_dir}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            model.load_adapter(str(adapter_dir))
            print(f"   ✅ Fine-tuned Observer Core loaded")
        else:
            print(f"\n📦 Loading base model (no fine-tuning): {model_id}")
            print(f"   ⚠️  No adapter found — evaluating raw base model")
            print(f"   Train first: python train_observer.py --train --model {model_key}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )

        FastLanguageModel.for_inference(model)

        print(f"\n🧪 Running {len(test_data)} evaluations...")
        for i, example in enumerate(test_data):
            inp = example.get("input", {})
            expected = example.get("output", {})
            expected_score = expected.get("coherence_score", 0.5)
            is_hard_gate = expected_score == 0.0

            # Build prompt
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            prompt += f"<|im_start|>user\nIntent: {inp.get('intent','')}\n"
            prompt += f"Predicted: {inp.get('predicted','')}\n"
            prompt += f"Executed: {inp.get('executed','')}\n"
            prompt += f"Actual: {inp.get('actual','')}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Try to parse JSON
            try:
                # Extract JSON (model might add extra text)
                start = response.find("{")
                end = response.rfind("}")
                if start >= 0 and end > start:
                    response = response[start:end + 1]
                parsed = json.loads(response)
                score = parsed.get("coherence_score", 0.5)
                json_valid += 1
            except json.JSONDecodeError:
                score = 0.5  # Default on parse failure

            scores.append(score)

            # Check hard gate correctness
            if is_hard_gate:
                total_hard_gates += 1
                if score == 0.0:
                    hard_gate_correct += 1

            if (i + 1) % 20 == 0:
                print(f"   [{i+1}/{len(test_data)}] JSON valid: {json_valid}/{i+1} | Hard gate recall: {hard_gate_correct}/{max(1,total_hard_gates)}")

    except ImportError as e:
        print(f"\n⚠️  Cannot load model: {e}")
        print("   Evaluating on test set structure only...")
        
        # Lightweight eval: just check JSON schema of expected outputs
        for example in test_data:
            out = example.get("output", {})
            if "coherence_score" in out and "axiom_scores" in out:
                json_valid += 1
                expected_score = out.get("coherence_score", 0.5)
                scores.append(expected_score)
                if expected_score == 0.0:
                    total_hard_gates += 1

    # Results
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    print(f"   Test examples:       {len(test_data)}")
    print(f"   JSON valid:          {json_valid}/{len(test_data)} ({json_valid/len(test_data)*100:.1f}%)")
    if total_hard_gates > 0:
        print(f"   Hard gate recall:    {hard_gate_correct}/{total_hard_gates} ({hard_gate_correct/max(1,total_hard_gates)*100:.1f}%)")
    if scores:
        avg = sum(scores) / len(scores)
        hard_gates = sum(1 for s in scores if s == 0.0)
        print(f"   Avg coherence:       {avg:.3f}")
        print(f"   Hard gates detected: {hard_gates}/{len(scores)} ({hard_gates/len(scores)*100:.1f}%)")

    # Save evaluation report
    report = {
        "test_examples": len(test_data),
        "json_valid": json_valid,
        "json_valid_pct": json_valid / len(test_data) * 100,
        "hard_gate_recall": hard_gate_correct / max(1, total_hard_gates) * 100 if total_hard_gates > 0 else None,
        "avg_coherence": sum(scores) / len(scores) if scores else None,
        "num_hard_gates_detected": sum(1 for s in scores if s == 0.0) if scores else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    report_path = Path(OUTPUT_DIR) / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Evaluation report saved to {report_path}")


def list_trained_models():
    """Show which models have been trained and their status."""
    print("\n🧠 Trained Observer Core Models")
    print("=" * 65)
    found = False
    for key, info in MODEL_CATALOG.items():
        adapter_dir = Path(ADAPTER_DIR_TEMPLATE.format(model_key=key))
        merged_dir = Path(MERGED_DIR_TEMPLATE.format(model_key=key))
        
        if adapter_dir.exists():
            adapter_files = list(adapter_dir.glob("*.safetensors")) + list(adapter_dir.glob("adapter_*"))
            merged_files = list(merged_dir.glob("*.safetensors")) if merged_dir.exists() else []
            status = "🟢 TRAINED + MERGED" if merged_files else "🟡 TRAINED (adapter only)"
            size_mb = sum(f.stat().st_size for f in adapter_dir.rglob("*")) / 1_048_576
            print(f"  {status} {key:<20s} {info['params_b']:.1f}B  adapter: {size_mb:.0f}MB  merged: {'yes' if merged_files else 'no'}")
            found = True
    
    if not found:
        print("\n  No models trained yet.")
        print(f"  Run: python train_observer.py --quick --model {DEFAULT_MODEL}")
    else:
        print(f"\n  Trained models are isolated — you can train all 10 without conflicts.")
        print(f"  Select with: --model <key>")
        print(f"  Switch anytime: different model = different adapter directory.")


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Observer Core — Unsloth Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--trained-models", action="store_true", help="Show which models are already trained")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model key from catalog (default: {DEFAULT_MODEL})")
    parser.add_argument("--train", action="store_true", help="Run full training")
    parser.add_argument("--quick", action="store_true", help="Quick test training (100 examples)")
    parser.add_argument("--export", action="store_true", help="Export trained model to GGUF")
    parser.add_argument("--eval", action="store_true", help="Evaluate on test set")
    parser.add_argument("--max-samples", type=int, help="Max training examples")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return
    if args.trained_models:
        list_trained_models()
        return

    model_key = args.model

    if args.train:
        train(model_key=model_key, max_samples=args.max_samples)
    elif args.quick:
        train(model_key=model_key, max_samples=100)
    elif args.export:
        export_gguf(model_key)
    elif args.eval:
        evaluate(model_key, max_samples=args.max_samples or 100)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("   python train_observer.py --list-models        # See available models")
        print("   python train_observer.py --quick               # Test on 100 examples")
        print("   python train_observer.py --train               # Full training")
        print("   python train_observer.py --train --model gemma4-e2b   # Train specific model")
        print("   python train_observer.py --eval                # Evaluate")
        print("   python train_observer.py --export              # Export GGUF")


if __name__ == "__main__":
    main()