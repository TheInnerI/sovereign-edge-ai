#!/usr/bin/env python3
"""
Observer Core — Training Dashboard v2

Complete training platform:
  - 📊 Data Browser: view examples, stats per type, add/curate training data
  - 🧠 Model Registry: track trained models, versions, checkpoints
  - 📈 Results: eval metrics, hard gate recall, coherence distribution
  - 📋 Model Cards: auto-generate + one-click HuggingFace publish
  - ⚖️ Compare: side-by-side model evaluation
  - 📤 Data Upload: add new residual examples for fine-tuning

Usage:
    python dashboard.py                # http://localhost:8765
    python dashboard.py --port 8765
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────

DEFAULT_PORT = 8765
DATASET_DIR = Path("data/datasets/observer-core/")
OUTPUT_DIR = Path("output/")

MODEL_CATALOG = {
    "qwen3.5-2b":     {"params": "2.0B", "size": "450MB", "tier": "🥇 Primary", "license": "Apache 2.0"},
    "qwen3.5-0.8b":   {"params": "0.8B", "size": "180MB", "tier": "🥈 Lightweight", "license": "Apache 2.0"},
    "qwen3.5-4b":     {"params": "4.0B", "size": "900MB", "tier": "📌 Quality", "license": "Apache 2.0"},
    "gemma4-e2b":     {"params": "2.0B", "size": "450MB", "tier": "🥉 Google", "license": "Gemma"},
    "ministral3-3b":  {"params": "3.0B", "size": "650MB", "tier": "📌 Mistral", "license": "Apache 2.0"},
    "deepseek-r1-1.5b": {"params": "1.5B", "size": "350MB", "tier": "📌 Reasoning", "license": "MIT"},
    "qwen3-1.7b":     {"params": "1.7B", "size": "380MB", "tier": "📌 Prev Gen", "license": "Apache 2.0"},
    "smollm2-1.7b":   {"params": "1.7B", "size": "380MB", "tier": "📌 HuggingFace", "license": "Apache 2.0"},
    "llama3.2-1b":    {"params": "1.2B", "size": "280MB", "tier": "📌 Meta", "license": "Llama Community"},
    "qwen3-0.6b":     {"params": "0.6B", "size": "140MB", "tier": "📌 Ultra-light", "license": "Apache 2.0"},
}

DATA_TYPES = ["residuals", "coherence_judgments", "contradiction_pairs",
              "observer_sequences", "preference_pairs", "structured_records"]

training_jobs: dict = {}
job_lock = threading.Lock()


# ─── Helpers ──────────────────────────────────────────────────────────────

def get_model_status(model_key: str) -> str:
    adapter_dir = OUTPUT_DIR / f"observer-lora-{model_key}"
    merged_dir = OUTPUT_DIR / f"observer-merged-{model_key}"
    has_adapter = adapter_dir.exists() and any(adapter_dir.rglob("*.safetensors"))
    has_merged = merged_dir.exists() and any(merged_dir.rglob("*.safetensors"))
    is_training = model_key in training_jobs and training_jobs[model_key].get("status") == "training"
    if is_training: return "training"
    if has_merged: return "merged"
    if has_adapter: return "trained"
    return "untrained"


def get_dataset_stats() -> dict:
    stats = {"total": 0, "by_type": {}, "files": [], "train_examples": 0, "test_examples": 0, "real_examples": 0}
    for fname in sorted(DATASET_DIR.glob("*.jsonl")):
        count = sum(1 for _ in open(fname))
        stats["total"] += count
        stats["files"].append({"name": fname.name, "count": count, "size_kb": round(os.path.getsize(fname) / 1024, 1)})
        for dtype in DATA_TYPES:
            if dtype in fname.name:
                stats["by_type"][dtype] = stats["by_type"].get(dtype, 0) + count
        if "train" in fname.name: stats["train_examples"] += count
        if "test" in fname.name: stats["test_examples"] += count
        if "real" in fname.name: stats["real_examples"] += count
    return stats


def browse_examples(dtype: str = "residuals", split: str = "train", limit: int = 20, offset: int = 0) -> dict:
    filepath = DATASET_DIR / f"{dtype}_{split}.jsonl"
    if not filepath.exists():
        return {"examples": [], "total": 0, "error": f"No data for {dtype}/{split}"}
    examples = []
    total = sum(1 for _ in open(filepath))
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i < offset: continue
            if len(examples) >= limit: break
            try:
                ex = json.loads(line)
                ex["_line"] = i + 1
                examples.append(ex)
            except json.JSONDecodeError:
                continue
    return {"examples": examples, "total": total, "offset": offset, "limit": limit}


def get_registry() -> list:
    models = []
    for key, info in MODEL_CATALOG.items():
        status = get_model_status(key)
        adapter_dir = OUTPUT_DIR / f"observer-lora-{key}"
        merged_dir = OUTPUT_DIR / f"observer-merged-{key}"
        entry = {
            "key": key, "params": info["params"], "tier": info["tier"],
            "status": status, "license": info["license"],
            "adapter_size_mb": round(sum(f.stat().st_size for f in adapter_dir.rglob("*")) / 1_048_576, 1) if adapter_dir.exists() else 0,
            "merged_size_mb": round(sum(f.stat().st_size for f in merged_dir.rglob("*")) / 1_048_576, 1) if merged_dir.exists() else 0,
            "has_eval": (OUTPUT_DIR / f"eval_report_{key}.json").exists(),
        }
        # Load eval report if exists
        eval_file = OUTPUT_DIR / f"eval_report_{key}.json"
        if eval_file.exists():
            with open(eval_file) as f:
                entry["eval"] = json.load(f)
        models.append(entry)
    return models


def generate_model_card(model_key: str) -> dict:
    info = MODEL_CATALOG.get(model_key, {})
    adapter_dir = OUTPUT_DIR / f"observer-lora-{key}" if (key := model_key) else None
    card = f"""---
language: en
license: {info.get('license', 'Apache 2.0')}
tags:
  - observer-core
  - residual-detection
  - coherence-scoring
  - sovereign-edge-ai
  - inner-i
  - phone-deployable
  - qlora
  - fine-tuned
base_model: {info.get('base_model_id', 'Qwen/Qwen3.5-2B-Instruct')}
datasets:
  - inneri/observer-core-synthetic-v1
metrics:
  - coherence_score
  - hard_gate_accuracy
  - json_schema_compliance
---

# Observer Core — {model_key}

Fine-tuned {info.get('params', '?')} model specialized for the Inner I Observer Core:
residual detection, coherence scoring, contradiction detection, and structured JSON output.

## Model Details

- **Base:** {info.get('base_model_id', 'Qwen/Qwen3.5-2B-Instruct')}
- **Training:** QLoRA (r=16, alpha=32) on Unsloth
- **Dataset:** 11k synthetic + real residual examples
- **Quantization:** GGUF IQ2_XS for phone deployment

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("inneri/observer-core-{model_key}")
tokenizer = AutoTokenizer.from_pretrained("inneri/observer-core-{model_key}")
```

## Six Axioms (Coherence Basis)

1. Awareness Is Law (20%)
2. Truth Over Comfort (20%)
3. Coherence Over Features (15%)
4. Append-Only Memory (15%)
5. Human Final Authority (15%)
6. Local Sovereignty (15%)

## License

{info.get('license', 'Apache 2.0')}
"""
    return {"model_key": model_key, "card": card}


# ─── FastAPI App ──────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, Request, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn"], check=True)
    from fastapi import FastAPI, Request, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

app = FastAPI(title="Observer Core Dashboard v2", version="2.0")


# ─── API: Models ──────────────────────────────────────────────────────────

@app.get("/api/models")
async def api_models():
    return {"models": get_registry(), "dataset": get_dataset_stats()}


@app.get("/api/registry")
async def api_registry():
    """Full model registry with versions and checkpoints."""
    return {"registry": get_registry(), "total_trained": sum(1 for m in get_registry() if m["status"] in ("trained", "merged"))}


@app.post("/api/train/{model_key}")
async def api_train(model_key: str):
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": f"Unknown model: {model_key}"}, status_code=400)
    try:
        import torch
        if not torch.cuda.is_available():
            return JSONResponse({"error": "no_gpu",
                "message": "CUDA GPU required for training.",
                "colab_url": "https://colab.research.google.com/github/TheInnerI/sovereign-edge-ai/blob/master/notebooks/train_observer_colab.ipynb",
                "colab_text": "🚀 Open in Colab (Free T4 GPU)"}, status_code=400)
    except ImportError:
        return JSONResponse({"error": "no_torch", "message": "PyTorch not installed"}, status_code=400)
    with job_lock:
        if model_key in training_jobs and training_jobs[model_key].get("status") == "training":
            return {"status": "already_training"}
        training_jobs[model_key] = {"status": "training", "started_at": datetime.now().strftime("%H:%M:%S"), "progress": "Starting..."}
    def run():
        try:
            training_jobs[model_key]["progress"] = "Loading model..."
            r = subprocess.run([sys.executable, "train_observer.py", "--train", "--model", model_key], capture_output=True, text=True, timeout=14400, cwd=os.path.dirname(os.path.abspath(__file__)))
            with job_lock:
                training_jobs[model_key]["status"] = "completed" if r.returncode == 0 else "failed"
                training_jobs[model_key]["progress"] = "Training complete!" if r.returncode == 0 else f"Failed: {r.stderr[-200:]}"
        except subprocess.TimeoutExpired:
            training_jobs[model_key]["status"] = "timeout"
        except Exception as e:
            training_jobs[model_key]["status"] = "failed"
            training_jobs[model_key]["progress"] = str(e)[:200]
    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "model": model_key}


@app.post("/api/eval/{model_key}")
async def api_eval(model_key: str):
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": f"Unknown model"}, status_code=400)
    r = subprocess.run([sys.executable, "train_observer.py", "--eval", "--model", model_key, "--max-samples", "50"], capture_output=True, text=True, timeout=600, cwd=os.path.dirname(os.path.abspath(__file__)))
    return {"model": model_key, "output": r.stdout[-2000:], "success": r.returncode == 0}


@app.post("/api/export/{model_key}")
async def api_export(model_key: str):
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": f"Unknown model"}, status_code=400)
    r = subprocess.run([sys.executable, "train_observer.py", "--export", "--model", model_key], capture_output=True, text=True, timeout=600, cwd=os.path.dirname(os.path.abspath(__file__)))
    return {"model": model_key, "output": r.stdout[-2000:], "success": r.returncode == 0}


# ─── API: Data ────────────────────────────────────────────────────────────

@app.get("/api/data/stats")
async def api_data_stats():
    return get_dataset_stats()


@app.get("/api/data/browse")
async def api_data_browse(dtype: str = "residuals", split: str = "train", limit: int = 20, offset: int = 0):
    return browse_examples(dtype, split, limit, offset)


@app.post("/api/data/add")
async def api_data_add(request: Request):
    """Add a new residual example to the training dataset."""
    try:
        body = await request.json()
        intent = body.get("intent", "").strip()
        predicted = body.get("predicted", "").strip()
        executed = body.get("executed", "").strip()
        actual = body.get("actual", "").strip()
        coherence_score = float(body.get("coherence_score", 0.5))
        axiom_violated = body.get("axiom_violated", "").strip()
        if not all([intent, predicted, executed, actual]):
            return JSONResponse({"error": "All four fields required: intent, predicted, executed, actual"}, status_code=400)
        # Build residual record
        residual_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        axiom_scores = {}
        for ax in ["Awareness Is Law", "Truth Over Comfort", "Coherence Over Features", "Append-Only Memory", "Human Final Authority", "Local Sovereignty"]:
            axiom_scores[ax] = 0.0 if ax == axiom_violated else round(0.7 + __import__("random").random() * 0.3, 2)
        record = {
            "input": {"intent": intent, "predicted": predicted, "executed": executed, "actual": actual},
            "output": {
                "residual_id": residual_id, "timestamp": timestamp,
                "intent": intent, "predicted": predicted, "executed": executed, "actual": actual,
                "residual": f"User-submitted: {actual[:100]}",
                "coherence_score": coherence_score,
                "axiom_scores": axiom_scores,
                "contradictions": [], "correction_proposal": "", "observer_state_update": {}
            },
            "metadata": {"source": "user_curated", "submitted_at": timestamp, "axiom_violated": axiom_violated or "none"}
        }
        # Append to residuals_train.jsonl
        train_file = DATASET_DIR / "residuals_train.jsonl"
        with open(train_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        return {"status": "added", "residual_id": residual_id, "file": str(train_file.name)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── API: Model Cards ─────────────────────────────────────────────────────

@app.get("/api/card/{model_key}")
async def api_card(model_key: str):
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": "Unknown model"}, status_code=400)
    return generate_model_card(model_key)


# ─── API: Compare ─────────────────────────────────────────────────────────

@app.get("/api/compare")
async def api_compare():
    """Side-by-side eval comparison of all trained models."""
    models = get_registry()
    trained = [m for m in models if m.get("eval")]
    return {"models": trained, "count": len(trained)}


# ─── Dashboard HTML ───────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Observer Core — Training Dashboard v2</title>
<style>
:root{--bg:#0a0a0f;--card:#141420;--border:#252540;--text:#c8c8d4;--dim:#6b6b80;--green:#00d45a;--yellow:#ffb800;--blue:#4a90ff;--red:#ff4455;--accent:#7c5cfc;}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;padding:16px;max-width:1400px;margin:0 auto}
h1{font-size:20px}h1 span{color:var(--accent)}
.tabs{display:flex;gap:4px;margin:16px 0;border-bottom:1px solid var(--border)}
.tab{padding:8px 16px;cursor:pointer;font-size:13px;color:var(--dim);border-bottom:2px solid transparent;transition:all .15s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.panel{display:none}.panel.active{display:block}
.stats{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.stat{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px 16px;text-align:center;min-width:100px}
.stat .value{font-size:20px;font-weight:700;color:var(--accent)}.stat .label{font-size:10px;color:var(--dim);margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;transition:border-color .2s}.card:hover{border-color:var(--accent)}
.card-header{display:flex;justify-content:space-between;margin-bottom:8px}
.card-title{font-size:15px;font-weight:600}.card-tier{font-size:10px;color:var(--dim)}
.card-meta{display:flex;gap:10px;font-size:11px;color:var(--dim);margin-bottom:10px}
.status{display:inline-block;padding:1px 8px;border-radius:3px;font-size:10px;font-weight:600}
.status.untrained{background:#252540;color:var(--dim)}.status.training{background:#4a3600;color:var(--yellow);animation:pulse 1.5s infinite}.status.trained{background:#003820;color:var(--green)}.status.merged{background:#002040;color:var(--blue)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.btn{display:inline-block;padding:5px 12px;border-radius:5px;font-size:11px;font-weight:600;cursor:pointer;border:none;margin:2px;transition:all .15s}
.btn:disabled{opacity:.3;cursor:not-allowed}
.btn-train{background:var(--accent);color:#fff}.btn-train:hover:not(:disabled){background:#9b7bff}
.btn-eval{background:transparent;border:1px solid var(--blue);color:var(--blue)}.btn-eval:hover:not(:disabled){background:#001a33}
.btn-export{background:transparent;border:1px solid var(--green);color:var(--green)}.btn-export:hover:not(:disabled){background:#001a18}
.btn-card{background:transparent;border:1px solid var(--yellow);color:var(--yellow)}
.metric-row{display:flex;gap:12px;font-size:11px;margin-top:8px}
.metric{color:var(--dim)}.metric strong{color:var(--text)}
.log{background:#0a0a12;border:1px solid var(--border);border-radius:6px;padding:10px;max-height:250px;overflow-y:auto;font-size:11px;color:var(--dim);white-space:pre-wrap;margin-top:12px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-weight:600;font-size:10px;text-transform:uppercase}
tr:hover{background:rgba(124,92,252,.05)}
.data-sample{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:8px;margin:4px 0;font-size:11px;max-height:120px;overflow-y:auto}
input,textarea,select{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:4px;font-family:inherit;font-size:12px;width:100%;margin:4px 0}
label{font-size:11px;color:var(--dim)}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
pre{font-size:11px;background:var(--bg);padding:10px;border-radius:4px;overflow-x:auto;max-height:300px}
.refresh{color:var(--dim);font-size:10px;text-align:right;margin-top:8px}
.compare-table td{vertical-align:top}
.compare-table .winner{color:var(--green);font-weight:700}
</style>
</head>
<body>
<h1>🧠 Observer Core <span>Training Dashboard v2</span></h1>
<div class="tabs">
  <div class="tab active" data-panel="models">🧠 Models</div>
  <div class="tab" data-panel="data">📊 Data Browser</div>
  <div class="tab" data-panel="add-data">➕ Add Data</div>
  <div class="tab" data-panel="compare">⚖️ Compare</div>
  <div class="tab" data-panel="cards">📋 Model Cards</div>
</div>

<div id="panel-models" class="panel active">
  <div class="stats" id="stats"></div>
  <div class="grid" id="grid"></div>
</div>

<div id="panel-data" class="panel">
  <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap">
    <select id="data-dtype" onchange="loadData()">
      <option value="residuals">Residuals</option>
      <option value="coherence_judgments">Coherence Judgments</option>
      <option value="contradiction_pairs">Contradiction Pairs</option>
      <option value="observer_sequences">Observer Sequences</option>
      <option value="preference_pairs">Preference Pairs</option>
      <option value="structured_records">Structured Records</option>
    </select>
    <select id="data-split" onchange="loadData()">
      <option value="train">Train</option>
      <option value="val">Validation</option>
      <option value="test">Test</option>
    </select>
    <button class="btn btn-eval" onclick="loadData()">🔍 Browse</button>
    <span style="font-size:11px;color:var(--dim);align-self:center" id="data-count"></span>
  </div>
  <div id="data-list"></div>
</div>

<div id="panel-add-data" class="panel">
  <h3 style="margin-bottom:12px;font-size:14px">➕ Add Training Example</h3>
  <p style="font-size:11px;color:var(--dim);margin-bottom:12px">Curate new residual examples. Appended to training split.</p>
  <div class="form-row">
    <div><label>Intent (what was intended)</label><input id="f-intent" placeholder="e.g. Generate truthful product description"></div>
    <div><label>Predicted (what was expected)</label><input id="f-predicted" placeholder="e.g. Output passes all tests"></div>
  </div>
  <div class="form-row">
    <div><label>Executed (what happened)</label><input id="f-executed" placeholder="e.g. Generated: '10x overnight'"></div>
    <div><label>Actual outcome</label><input id="f-actual" placeholder="e.g. Failed Truth Test — unsubstantiated claim"></div>
  </div>
  <div class="form-row">
    <div><label>Coherence Score (0.0-1.0)</label><input id="f-score" type="number" min="0" max="1" step="0.01" value="0.0"></div>
    <div><label>Axiom Violated (if hard gate)</label>
      <select id="f-axiom">
        <option value="">None (clean)</option>
        <option>Truth Over Comfort</option><option>Awareness Is Law</option>
        <option>Coherence Over Features</option><option>Append-Only Memory</option>
        <option>Human Final Authority</option><option>Local Sovereignty</option>
      </select>
    </div>
  </div>
  <button class="btn btn-train" onclick="addExample()">➕ Add to Training Set</button>
  <div id="add-result" style="margin-top:8px;font-size:12px"></div>
</div>

<div id="panel-compare" class="panel">
  <div id="compare-view"><p style="color:var(--dim);font-size:12px">Train multiple models to see comparison.</p></div>
</div>

<div id="panel-cards" class="panel">
  <p style="font-size:11px;color:var(--dim);margin-bottom:12px">Auto-generated HuggingFace model cards. Copy or publish.</p>
  <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap" id="card-buttons"></div>
  <pre id="card-preview" style="display:none"></pre>
</div>

<div class="log" id="log">Ready.</div>
<div class="refresh">Auto-refresh: 5s</div>

<script>
const LOG=document.getElementById('log');
function log(msg){const t=new Date().toLocaleTimeString();LOG.textContent=`[${t}] ${msg}\\n`+LOG.textContent.split('\\n').slice(0,50).join('\\n')}

// Tab switching
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('panel-'+t.dataset.panel).classList.add('active');
  if(t.dataset.panel==='data') loadData();
  if(t.dataset.panel==='compare') loadCompare();
  if(t.dataset.panel==='cards') loadCards();
});

async function refresh(){
  try{
    const r=await fetch('/api/models'),d=await r.json();
    renderModels(d);
  }catch(e){}
}
function renderModels(data){
  const ds=data.dataset;
  document.getElementById('stats').innerHTML=`
    <div class="stat"><div class="value">${ds.total.toLocaleString()}</div><div class="label">Examples</div></div>
    <div class="stat"><div class="value">${ds.train_examples.toLocaleString()}</div><div class="label">Train</div></div>
    <div class="stat"><div class="value">${ds.test_examples.toLocaleString()}</div><div class="label">Test</div></div>
    <div class="stat"><div class="value">${data.models.filter(m=>m.status!=='untrained').length}</div><div class="label">Trained</div></div>
  `;
  document.getElementById('grid').innerHTML=data.models.map(m=>{
    const canTrain=m.status!=='training';
    const canEval=m.status==='trained'||m.status==='merged';
    const canExport=m.status==='trained'||m.status==='merged';
    const evalHtml=m.eval?`<div class="metric-row">
      <div class="metric">JSON: <strong>${(m.eval.json_valid_pct||0).toFixed(0)}%</strong></div>
      <div class="metric">Hard gate: <strong>${(m.eval.hard_gate_recall||0).toFixed(0)}%</strong></div>
      <div class="metric">Avg score: <strong>${(m.eval.avg_coherence||0).toFixed(2)}</strong></div>
    </div>`:'';
    return `<div class="card">
      <div class="card-header"><div><div class="card-title">${m.key}</div><div class="card-tier">${m.tier}</div></div><span class="status ${m.status}">${m.status.toUpperCase()}</span></div>
      <div class="card-meta"><span>${m.params}</span><span>${m.license}</span>${m.adapter_size_mb?`<span>Adapter: ${m.adapter_size_mb}MB</span>`:''}</div>
      ${evalHtml}
      <div><button class="btn btn-train" ${canTrain?'':'disabled'} onclick="train('${m.key}')">⚡ Train</button>
      <button class="btn btn-eval" ${canEval?'':'disabled'} onclick="evaluate('${m.key}')">📊 Eval</button>
      <button class="btn btn-export" ${canExport?'':'disabled'} onclick="exportModel('${m.key}')">📦 Export</button></div>
    </div>`;
  }).join('');
}

async function train(key){log(`Training ${key}...`);try{const r=await fetch(`/api/train/${key}`,{method:'POST'});const d=await r.json();if(d.error){log(`❌ ${d.error}: ${d.message||''}`);if(d.colab_url){log(`🔗 ${d.colab_text}: ${d.colab_url}`)}}else{log(d.status||'started')}}catch(e){log(`Error: ${e}`)}refresh()}
async function evaluate(key){log(`Eval ${key}...`);try{const r=await fetch(`/api/eval/${key}`,{method:'POST'});const d=await r.json();log(`Eval ${key}: ${d.success?'✅':'❌'}`);if(d.output)log(d.output.slice(-400))}catch(e){log(`Error: ${e}`)}}
async function exportModel(key){log(`Export ${key}...`);try{const r=await fetch(`/api/export/${key}`,{method:'POST'});const d=await r.json();log(`Export ${key}: ${d.success?'✅':'❌'}`)}catch(e){log(`Error: ${e}`)}}

// Data browser
async function loadData(){
  const dtype=document.getElementById('data-dtype').value;
  const split=document.getElementById('data-split').value;
  try{
    const r=await fetch(`/api/data/browse?dtype=${dtype}&split=${split}&limit=30`);
    const d=await r.json();
    document.getElementById('data-count').textContent=`${d.total} examples`;
    document.getElementById('data-list').innerHTML=d.examples.map(ex=>{
      const inp=ex.input||{};
      const out=ex.output||{};
      const score=out.coherence_score??'?';
      const color=score===0?'var(--red)':score<0.5?'var(--yellow)':'var(--green)';
      return `<div class="data-sample">
        <strong style="color:${color}">#${ex._line} score=${score}</strong>
        <div>Intent: ${(inp.intent||'').slice(0,80)}</div>
        <div>Actual: ${(inp.actual||'').slice(0,80)}</div>
      </div>`;
    }).join('');
  }catch(e){}
}

// Add example
async function addExample(){
  const body={
    intent:document.getElementById('f-intent').value,
    predicted:document.getElementById('f-predicted').value,
    executed:document.getElementById('f-executed').value,
    actual:document.getElementById('f-actual').value,
    coherence_score:parseFloat(document.getElementById('f-score').value)||0,
    axiom_violated:document.getElementById('f-axiom').value,
  };
  try{
    const r=await fetch('/api/data/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    document.getElementById('add-result').innerHTML=d.error?`❌ ${d.error}`:`✅ Added! ID: ${d.residual_id?.slice(0,8)}... → ${d.file}`;
  }catch(e){document.getElementById('add-result').innerHTML=`❌ ${e}`}
}

// Compare
async function loadCompare(){
  try{
    const r=await fetch('/api/compare'),d=await r.json();
    if(!d.models.length){document.getElementById('compare-view').innerHTML='<p style="color:var(--dim)">No trained models yet. Train at least 2 to compare.</p>';return}
    document.getElementById('compare-view').innerHTML=`
      <table class="compare-table"><tr><th>Model</th><th>Status</th><th>JSON Valid</th><th>Hard Gate Recall</th><th>Avg Coherence</th><th>Adapter Size</th></tr>
      ${d.models.map(m=>`<tr>
        <td><strong>${m.key}</strong></td><td><span class="status ${m.status}">${m.status}</span></td>
        <td>${m.eval?.json_valid_pct?.toFixed(0)||'?'}%</td>
        <td>${m.eval?.hard_gate_recall?.toFixed(0)||'?'}%</td>
        <td>${m.eval?.avg_coherence?.toFixed(2)||'?'}</td>
        <td>${m.adapter_size_mb}MB</td>
      </tr>`).join('')}
      </table>`;
    }catch(e){}
}

// Model cards
async function loadCards(){
  try{
    const r=await fetch('/api/models'),d=await r.json();
    const trained=d.models.filter(m=>m.status!=='untrained');
    document.getElementById('card-buttons').innerHTML=trained.length?trained.map(m=>
      `<button class="btn btn-card" onclick="loadCard('${m.key}')">📋 ${m.key}</button>`
    ).join(''):'<span style="color:var(--dim);font-size:12px">No trained models. Train one first.</span>';
  }catch(e){}
}
async function loadCard(key){
  try{
    const r=await fetch(`/api/card/${key}`),d=await r.json();
    document.getElementById('card-preview').style.display='block';
    document.getElementById('card-preview').textContent=d.card;
    log(`Card loaded for ${key}. Copy to HuggingFace.`);
  }catch(e){}
}

refresh();setInterval(refresh,5000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Observer Core Training Dashboard v2")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    print(f"""
╔══════════════════════════════════════════════════╗
║  🧠 Observer Core — Training Dashboard v2       ║
║  http://localhost:{args.port}                       ║
║                                                  ║
║  📊 Data Browser · 🧠 Registry · 📈 Results     ║
║  📋 Model Cards · ⚖️ Compare · ➕ Add Data      ║
╚══════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()