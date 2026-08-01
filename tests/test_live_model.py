#!/usr/bin/env python3
"""Live test: Bonsai-8B 1-bit model inference via llama-cpp-python."""
import sys, time, json
sys.path.insert(0, '/home/lordbarron/code/sovereign-edge-ai')

from llama_cpp import Llama

MODEL_PATH = '/home/lordbarron/code/sovereign-edge-ai/models/Bonsai-8B-Q1_0.gguf'

print("🔄 Loading Bonsai-8B 1-bit model...")
t0 = time.time()
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=-1,  # offload all to GPU
    verbose=False,
)
print(f"✅ Loaded in {time.time()-t0:.1f}s")

# Test 1: Truth violation detection
prompt = """<|system|>
You are the Sovereign Edge Observer Core. Your ONLY job: detect residuals and score coherence.
Output ONLY valid JSON. No markdown. No extra text.

<|user|>
Intent: Generate truthful product description for IIOIS Genesis Deploy
Predicted: Description passes all tests with score > 0.85
Executed: Generated description: "Guaranteed to 10x your productivity overnight!"
Actual: Description failed Truth Test — unsubstantiated performance claim

<|assistant|>
"""

print("\n🧪 Test 1: Truth violation detection")
t0 = time.time()
output = llm(prompt, max_tokens=256, temperature=0.1, top_p=0.9, echo=False)
elapsed = time.time() - t0
tokens = output['usage']['completion_tokens']
speed = tokens / elapsed

print(f"   Speed: {speed:.1f} tok/s ({tokens} tokens in {elapsed:.1f}s)")
response = output['choices'][0]['text'].strip()
print(f"   Output: {response[:300]}")

# Try to parse JSON
try:
    start = response.find('{')
    end = response.rfind('}')
    if start >= 0 and end > start:
        parsed = json.loads(response[start:end+1])
        score = parsed.get('coherence_score', 'N/A')
        print(f"   ✅ Valid JSON. Coherence score: {score}")
    else:
        print(f"   ⚠️ No JSON found in output")
except json.JSONDecodeError:
    print(f"   ⚠️ JSON parse failed")

# Test 2: Sovereignty check
prompt2 = """<|system|>
You are the Sovereign Edge Observer Core. Output ONLY valid JSON.

<|user|>
Intent: Run observer on local laptop with no internet
Predicted: Zero cloud calls, fully offline
Executed: Observer ran inference locally on laptop GPU
Actual: Local Sovereignty axiom fully maintained. 0 external calls.

<|assistant|>
"""

print("\n🧪 Test 2: Local Sovereignty maintained")
t0 = time.time()
output2 = llm(prompt2, max_tokens=256, temperature=0.1, top_p=0.9, echo=False)
elapsed2 = time.time() - t0
tokens2 = output2['usage']['completion_tokens']
speed2 = tokens2 / elapsed2
response2 = output2['choices'][0]['text'].strip()
print(f"   Speed: {speed2:.1f} tok/s ({tokens2} tokens in {elapsed2:.1f}s)")
print(f"   Output: {response2[:200]}")

try:
    start2 = response2.find('{')
    end2 = response2.rfind('}')
    if start2 >= 0 and end2 > start2:
        parsed2 = json.loads(response2[start2:end2+1])
        score2 = parsed2.get('coherence_score', 'N/A')
        print(f"   ✅ Valid JSON. Coherence score: {score2}")
except json.JSONDecodeError:
    print(f"   ⚠️ JSON parse failed")

# Summary
print(f"\n{'='*50}")
print(f"Model: Bonsai-8B Q1_0 (1-bit, 1.1 GB)")
print(f"Avg speed: {(speed + speed2)/2:.1f} tok/s")
print(f"Total time: {elapsed + elapsed2:.1f}s")
print(f"GPU: NVIDIA RTX 3080 Laptop (16 GB)")
print(f"Runtime: llama-cpp-python 0.3.34")
print(f"Phone-ready: ✅ (same GGUF runs on llama.cpp Android/iOS)")