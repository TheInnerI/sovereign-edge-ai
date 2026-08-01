#!/usr/bin/env python3
"""
Observer Core — Training Dashboard

Single-file FastAPI app with embedded web UI for:
  - Viewing all 10 models + training status
  - Starting/stopping training runs
  - Evaluating trained models
  - Exporting to GGUF
  - Live progress monitoring

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
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────

DEFAULT_PORT = 8765
DATASET_DIR = "data/datasets/observer-core/"
OUTPUT_DIR = Path("output/")

# Model catalog (mirrors train_observer.py)
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

# Training jobs state
training_jobs: dict = {}       # {model_key: {status, pid, started_at, progress}}
job_lock = threading.Lock()


def get_model_status(model_key: str) -> dict:
    """Determine model training status."""
    adapter_dir = OUTPUT_DIR / f"observer-lora-{model_key}"
    merged_dir = OUTPUT_DIR / f"observer-merged-{model_key}"
    
    has_adapter = adapter_dir.exists() and any(adapter_dir.rglob("*.safetensors"))
    has_merged = merged_dir.exists() and any(merged_dir.rglob("*.safetensors"))
    
    # Check if currently training
    is_training = model_key in training_jobs and training_jobs[model_key].get("status") == "training"
    
    if is_training:
        return "training"
    elif has_merged:
        return "merged"
    elif has_adapter:
        return "trained"
    return "untrained"


def get_dataset_stats() -> dict:
    """Quick dataset stats."""
    train_file = Path(DATASET_DIR) / "residuals_train.jsonl"
    test_file = Path(DATASET_DIR) / "residuals_test.jsonl"
    
    stats = {
        "train": sum(1 for _ in open(train_file)) if train_file.exists() else 0,
        "test": sum(1 for _ in open(test_file)) if test_file.exists() else 0,
        "total": 0,
    }
    
    # Count all JSONL files
    for f in Path(DATASET_DIR).glob("*.jsonl"):
        stats["total"] += sum(1 for _ in open(f))
    
    return stats


# ─── FastAPI App ──────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("Installing FastAPI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"], check=True)
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

app = FastAPI(title="Observer Core Dashboard", version="1.0")


# ─── API Endpoints ────────────────────────────────────────────────────────

@app.get("/api/models")
async def api_models():
    """Get all models with status."""
    models = []
    for key, info in MODEL_CATALOG.items():
        status = get_model_status(key)
        job = training_jobs.get(key, {})
        models.append({
            "key": key,
            "params": info["params"],
            "size": info["size"],
            "tier": info["tier"],
            "license": info["license"],
            "status": status,
            "training_progress": job.get("progress", ""),
            "started_at": job.get("started_at", ""),
        })
    return {"models": models, "dataset": get_dataset_stats()}


@app.post("/api/train/{model_key}")
async def api_train(model_key: str):
    """Start training a model."""
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": f"Unknown model: {model_key}"}, status_code=400)
    
    with job_lock:
        if model_key in training_jobs and training_jobs[model_key].get("status") == "training":
            return {"status": "already_training", "message": f"{model_key} is already training"}
        
        training_jobs[model_key] = {
            "status": "training",
            "started_at": datetime.now().strftime("%H:%M:%S"),
            "progress": "Starting...",
            "pid": None,
        }
    
    # Launch training in background
    def run_training():
        try:
            training_jobs[model_key]["progress"] = "Loading model..."
            result = subprocess.run(
                [sys.executable, "train_observer.py", "--train", "--model", model_key],
                capture_output=True, text=True, timeout=14400,  # 4 hour timeout
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            with job_lock:
                if result.returncode == 0:
                    training_jobs[model_key]["status"] = "completed"
                    training_jobs[model_key]["progress"] = "Training complete!"
                else:
                    training_jobs[model_key]["status"] = "failed"
                    training_jobs[model_key]["progress"] = f"Failed: {result.stderr[-200:]}"
        except subprocess.TimeoutExpired:
            training_jobs[model_key]["status"] = "timeout"
            training_jobs[model_key]["progress"] = "Training timed out (4h limit)"
        except Exception as e:
            training_jobs[model_key]["status"] = "failed"
            training_jobs[model_key]["progress"] = str(e)[:200]
    
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    
    return {"status": "started", "model": model_key, "message": f"Training {model_key} started"}


@app.post("/api/eval/{model_key}")
async def api_eval(model_key: str):
    """Evaluate a trained model."""
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": f"Unknown model: {model_key}"}, status_code=400)
    
    result = subprocess.run(
        [sys.executable, "train_observer.py", "--eval", "--model", model_key, "--max-samples", "50"],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    
    return {
        "model": model_key,
        "output": result.stdout[-2000:],
        "success": result.returncode == 0,
    }


@app.post("/api/export/{model_key}")
async def api_export(model_key: str):
    """Export a trained model to GGUF."""
    if model_key not in MODEL_CATALOG:
        return JSONResponse({"error": f"Unknown model: {model_key}"}, status_code=400)
    
    result = subprocess.run(
        [sys.executable, "train_observer.py", "--export", "--model", model_key],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    
    return {
        "model": model_key,
        "output": result.stdout[-2000:],
        "success": result.returncode == 0,
    }


# ─── Web Dashboard (single-file HTML) ─────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Observer Core — Training Dashboard</title>
<style>
  :root {
    --bg: #0a0a0f;
    --card: #141420;
    --border: #252540;
    --text: #c8c8d4;
    --dim: #6b6b80;
    --green: #00d45a;
    --yellow: #ffb800;
    --blue: #4a90ff;
    --red: #ff4455;
    --accent: #7c5cfc;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'SF Mono','Fira Code',monospace; padding:24px; max-width:1200px; margin:0 auto; }
  h1 { font-size:22px; margin-bottom:4px; }
  h1 span { color:var(--accent); }
  .subtitle { color:var(--dim); font-size:13px; margin-bottom:24px; }
  .stats { display:flex; gap:16px; margin-bottom:24px; }
  .stat { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px 20px; text-align:center; }
  .stat .value { font-size:24px; font-weight:700; color:var(--accent); }
  .stat .label { font-size:11px; color:var(--dim); margin-top:4px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; transition:border-color 0.2s; }
  .card:hover { border-color:var(--accent); }
  .card-header { display:flex; justify-content:space-between; align-items:start; margin-bottom:12px; }
  .card-title { font-size:16px; font-weight:600; }
  .card-tier { font-size:11px; color:var(--dim); }
  .card-meta { display:flex; gap:12px; font-size:12px; color:var(--dim); margin-bottom:12px; }
  .status { display:inline-block; padding:2px 10px; border-radius:4px; font-size:11px; font-weight:600; }
  .status.untrained { background:#252540; color:var(--dim); }
  .status.training { background:#4a3600; color:var(--yellow); animation:pulse 1.5s infinite; }
  .status.trained { background:#003820; color:var(--green); }
  .status.merged { background:#002040; color:var(--blue); }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
  .btn { display:inline-block; padding:6px 14px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; border:none; margin:2px; transition:all 0.15s; }
  .btn:disabled { opacity:0.3; cursor:not-allowed; }
  .btn-train { background:var(--accent); color:#fff; }
  .btn-train:hover:not(:disabled) { background:#9b7bff; }
  .btn-eval { background:transparent; border:1px solid var(--blue); color:var(--blue); }
  .btn-eval:hover:not(:disabled) { background:#001a33; }
  .btn-export { background:transparent; border:1px solid var(--green); color:var(--green); }
  .btn-export:hover:not(:disabled) { background:#001a18; }
  .progress { font-size:11px; color:var(--dim); margin-top:8px; font-style:italic; }
  .log { background:#0a0a12; border:1px solid var(--border); border-radius:6px; padding:12px; margin-top:24px; max-height:300px; overflow-y:auto; font-size:12px; color:var(--dim); white-space:pre-wrap; }
  .refresh { color:var(--dim); font-size:11px; text-align:right; margin-top:8px; }
</style>
</head>
<body>
  <h1>🧠 Observer Core <span>Training Dashboard</span></h1>
  <p class="subtitle">10 models · QLoRA fine-tuning · phone-deployable</p>
  
  <div class="stats" id="stats"></div>
  <div class="grid" id="grid"></div>
  <div class="refresh">Auto-refresh: 5s</div>
  <div class="log" id="log">Ready. Select a model to begin.</div>

<script>
const LOG = document.getElementById('log');
const GRID = document.getElementById('grid');
const STATS = document.getElementById('stats');

function log(msg) {
  const t = new Date().toLocaleTimeString();
  LOG.textContent = `[${t}] ${msg}\\n` + LOG.textContent.split('\\n').slice(0, 50).join('\\n');
}

async function refresh() {
  try {
    const r = await fetch('/api/models');
    const data = await r.json();
    render(data);
  } catch(e) {
    log('Connection error — is the dashboard running?');
  }
}

function render(data) {
  // Stats
  const ds = data.dataset;
  STATS.innerHTML = `
    <div class="stat"><div class="value">${ds.total.toLocaleString()}</div><div class="label">Dataset Examples</div></div>
    <div class="stat"><div class="value">${ds.train.toLocaleString()}</div><div class="label">Training Split</div></div>
    <div class="stat"><div class="value">${ds.test.toLocaleString()}</div><div class="label">Test Split</div></div>
    <div class="stat"><div class="value">${data.models.filter(m=>m.status!=='untrained').length}</div><div class="label">Models Trained</div></div>
  `;

  // Cards
  GRID.innerHTML = data.models.map(m => {
    const statusClass = m.status;
    const statusLabel = m.status.toUpperCase();
    const canTrain = m.status !== 'training';
    const canEval = m.status === 'trained' || m.status === 'merged';
    const canExport = m.status === 'trained' || m.status === 'merged';
    const progress = m.training_progress ? `<div class="progress">${m.training_progress}</div>` : '';
    
    return `
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">${m.key}</div>
            <div class="card-tier">${m.tier}</div>
          </div>
          <span class="status ${statusClass}">${statusLabel}</span>
        </div>
        <div class="card-meta">
          <span>${m.params}</span>
          <span>~${m.size}</span>
          <span>${m.license}</span>
        </div>
        <div>
          <button class="btn btn-train" ${canTrain?'':'disabled'} onclick="train('${m.key}')">⚡ Train</button>
          <button class="btn btn-eval" ${canEval?'':'disabled'} onclick="evaluate('${m.key}')">📊 Eval</button>
          <button class="btn btn-export" ${canExport?'':'disabled'} onclick="exportModel('${m.key}')">📦 Export</button>
        </div>
        ${progress}
      </div>`;
  }).join('');
}

async function train(key) {
  log(`Starting training: ${key}...`);
  try {
    const r = await fetch(`/api/train/${key}`, {method:'POST'});
    const d = await r.json();
    log(d.message || JSON.stringify(d));
  } catch(e) { log(`Error: ${e}`); }
  refresh();
}

async function evaluate(key) {
  log(`Evaluating: ${key}...`);
  try {
    const r = await fetch(`/api/eval/${key}`, {method:'POST'});
    const d = await r.json();
    log(`Eval ${key}: ${d.success?'PASS':'FAIL'}`);
    if (d.output) log(d.output.slice(-500));
  } catch(e) { log(`Error: ${e}`); }
}

async function exportModel(key) {
  log(`Exporting: ${key}...`);
  try {
    const r = await fetch(`/api/export/${key}`, {method:'POST'});
    const d = await r.json();
    log(`Export ${key}: ${d.success?'PASS':'FAIL'}`);
  } catch(e) { log(`Error: ${e}`); }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Observer Core Training Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════╗
║  🧠 Observer Core — Training Dashboard          ║
║  http://localhost:{args.port}                       ║
║                                                  ║
║  10 models · QLoRA · phone-deployable            ║
║  Click "Train" to fine-tune a model              ║
╚══════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()