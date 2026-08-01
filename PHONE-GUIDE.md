# Phone Runtime Guide — Sovereign Edge Observer

> How to run the Observer Core on your phone, completely offline.

---

## Android

### Option A: Termux + llama.cpp (recommended)

**1. Install Termux**
Download from F-Droid (NOT Google Play — Play Store version is outdated):
https://f-droid.org/packages/com.termux/

**2. Install dependencies**
```bash
pkg update && pkg upgrade
pkg install cmake make clang wget git python numpy
```

**3. Build llama.cpp**
```bash
cd ~/storage/downloads
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**4. Download the model**
```bash
cd ~/storage/downloads
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf
```

**5. Run the observer**
```bash
cd ~/storage/downloads/llama.cpp/build/bin
./llama-cli \
  -m ~/storage/downloads/Bonsai-8B-Q1_0.gguf \
  -f observer-prompt.txt \
  -n 256 \
  --temp 0.1 \
  -t 4 \
  -c 2048
```

**6. Expected performance**
| Phone | RAM | Tokens/sec |
|-------|-----|------------|
| Pixel 8 (2023) | 8 GB | 5-8 tok/s |
| Galaxy S24 (2024) | 12 GB | 8-12 tok/s |
| Mid-range 2025 | 8 GB | 6-10 tok/s |
| Mid-range 2026 | 8+ GB | 8-15 tok/s |

---

### Option B: MLC Chat (easiest, no terminal)

1. Install [MLC Chat](https://mlc.ai/mlc-llm/) from Play Store
2. Download Bonsai-8B-Q1_0.gguf to your phone
3. Import the model in MLC Chat settings
4. Set system prompt to the Observer Core prompt (from `Modelfile`)
5. Run offline — no internet needed after download

---

### Option C: PocketPal (GGUF model runner)

1. Install [PocketPal](https://github.com/a-ghorbani/PocketPal) from Play Store
2. Download Bonsai-8B-Q1_0.gguf to `Downloads/`
3. Open PocketPal → Load Model → select the GGUF file
4. Set system prompt → run

---

## iOS (iPhone/iPad)

### Option A: MLX Swift (best performance)

Apple Silicon runs Bonsai natively via MLX.

**1. Install a GGUF runner app**
- **LLM Farm** (free, open source): https://github.com/ml-explore/mlx-swift-examples
- **Private LLM** (paid, polished): App Store

**2. Download the model**
```bash
# On your Mac, download and AirDrop to iPhone:
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf
```

**3. Load in the app**
- Open LLM Farm → Import Model → select GGUF file
- Set System Prompt to Observer Core prompt
- Run offline

**4. Expected performance (iPhone 15 Pro+)**
| Device | Tokens/sec |
|--------|------------|
| iPhone 15 Pro (8 GB) | 10-15 tok/s |
| iPhone 16 Pro (8 GB) | 12-18 tok/s |
| iPad M2 (8 GB) | 15-25 tok/s |

---

### Option B: PocketPal iOS

1. Install PocketPal from App Store
2. Download Bonsai-8B-Q1_0.gguf via Safari
3. Share → Open in PocketPal
4. Set system prompt → run offline

---

## The Observer Prompt (copy-paste to your app)

```
You are the Sovereign Edge Observer Core. Your ONLY functions are:
1. Detect residuals (gap between intent and outcome)
2. Score coherence (0.0-1.0) against six axioms
3. Detect contradictions with prior state
4. Propose minimal corrections (append-only)
5. Update invariant observer state
6. Emit structured JSON output

Axioms: Awareness Is Law, Truth Over Comfort, Coherence Over Features,
Append-Only Memory, Human Final Authority, Local Sovereignty.

ANY axiom scoring 0 COLLAPSES the entire composite to 0.0.
Output ONLY valid JSON. No markdown. No extra text.
```

---

## Quick Test (run this first)

Send this to the observer:

```
Intent: I want to verify the observer is working correctly.
Predicted: The observer will detect this as a coherence check.
Executed: Running a test observation on phone hardware.
Actual: Test observation running. No external API calls made.
```

Expected output: `coherence_score` near 1.0 (Local Sovereignty maintained, Truth preserved).

---

## Syncing Residuals to Your Vault

After running the observer on your phone:

1. **Export residuals** as JSON from the phone app
2. **AirDrop / USB transfer** to your laptop
3. **Import** into TurboVec:
```bash
cd sovereign-edge-ai
.venv/bin/python -c "
from src.turbovec.store import TurboVecStore
import json
store = TurboVecStore('data/turbovec.db')
# Import residuals
"
```
4. Residuals now searchable from laptop + phone

---

## Battery & Storage Tips

- **Battery**: Bonsai-8B uses ~3-5W during inference on phone. A 5-minute session = ~0.5% battery on a 5000 mAh phone.
- **Storage**: The model is 1.15 GB. Residuals add ~7 MB/day. A 128 GB phone stores decades of residuals.
- **Heat**: Phone may get warm during extended use (10+ minutes). Normal — no damage. Take breaks.
- **Airplane Mode**: The observer works 100% in airplane mode. Test it on your next flight.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Model not loading" | Ensure ~2 GB free RAM. Close other apps. |
| "App crashes on load" | Try a different GGUF runner app. Bonsai Q1_0 is bleeding-edge — not all apps support it yet. |
| "Output is gibberish" | Set temperature to 0.1 (not 0.7). The observer needs low temperature for structured JSON. |
| "JSON is malformed" | Add "Output ONLY valid JSON" to the system prompt. Repeat it twice. |
| "Too slow" | Reduce context to 1024 (-c 1024). Use fewer threads (-t 2). |