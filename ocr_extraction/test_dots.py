import os
import time
import io
import base64
import requests
import subprocess
import sys
from PIL import Image
from openai import OpenAI
import ollama

INPUT_FILE = "DASA-Statement-1.png"

print("🚀 Starting independent dots.mocr + Gemma test...")

# 1. Load Image
try:
    img = Image.open(INPUT_FILE).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    print(f"✅ Image '{INPUT_FILE}' loaded.")
except Exception as e:
    print(f"❌ Failed to load image: {e}")
    sys.exit(1)

# 2. Start vLLM (WITH LOGS EXPOSED)
print("\n⏳ Spinning up vLLM server...")
print("   (If it crashes, we will print the exact reason below)")

# Notice: stdout and stderr are now captured using PIPE instead of DEVNULL
vllm_proc = subprocess.Popen([
    "uv", "run", "vllm", "serve", "rednote-hilab/dots.mocr",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "0.6", 
    "--chat-template-content-format", "string",
    "--trust-remote-code",
    "--port", "8000",
    "--max-model-len", "8192"
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# 3. Poll for health
vllm_ready = False
start_time = time.time()

while time.time() - start_time < 120:
    # Check if the process died prematurely
    if vllm_proc.poll() is not None:
        print("\n❌ FATAL: vLLM process crashed instantly! Here are the logs:")
        print("-" * 50)
        print(vllm_proc.stdout.read())  # Print the exact crash reason
        print("-" * 50)
        sys.exit(1)

    try:
        if requests.get("http://localhost:8000/health").status_code == 200:
            vllm_ready = True
            break
    except requests.exceptions.ConnectionError:
        time.sleep(2)
        print(".", end="", flush=True)

if not vllm_ready:
    print("\n❌ vLLM timed out after 120 seconds. Dumping logs:")
    print("-" * 50)
    print(vllm_proc.stdout.read())
    print("-" * 50)
    vllm_proc.terminate()
    sys.exit(1)

print("\n\n🟢 vLLM is alive! Extracting raw text with dots.mocr...")

# 4. Stage 1: OCR Extraction
try:
    client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    resp = client.chat.completions.create(
        model="rednote-hilab/dots.mocr",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "text", "text": "<|img|><|imgpad|><|endofimg|>Extract the text content from this image."},
            ]
        }],
        max_completion_tokens=4096,
        temperature=0.1
    )
    raw_ocr = resp.choices[0].message.content
    print("\n--- RAW OCR RESULT ---")
    print(raw_ocr[:500] + "...\n[Truncated for display]")
    print("----------------------\n")
except Exception as e:
    print(f"❌ OpenAI client request failed: {e}")
    vllm_proc.terminate()
    sys.exit(1)

# 5. Stage 2: JSON Structuring via Ollama
print("🧠 Structuring text into JSON using Ollama (gemma3n:e2b)...")
FINANCE_SYSTEM = """You are a financial extraction expert.
Given raw OCR text from a bank statement, return ONLY a JSON object with this exact schema:
{
  "summary": { "opening_balance": float, "closing_balance": float, "total_deposits": float, "total_withdrawals": float },
  "transactions": [ {"date": "string", "description": "string", "amount": float, "type": "debit or credit"} ]
}
Rules: use positive numbers for both debits and credits. Output raw JSON only."""

try:
    msgs = [
        {"role": "system", "content": FINANCE_SYSTEM},
        {"role": "user", "content": f"Bank statement text:\n\n{raw_ocr}"},
    ]
    ollama_resp = ollama.chat(model="gemma3n:e2b", messages=msgs, format="json", options={"temperature": 0})
    structured_json = ollama_resp["message"]["content"]
    
    print("\n✅ --- FINAL STRUCTURED JSON ---")
    print(structured_json)
    print("--------------------------------")

except Exception as e:
    print(f"❌ Ollama structuring failed: {e}")

finally:
    # 6. Cleanup
    print("\n🛑 Shutting down vLLM server to free memory...")
    vllm_proc.terminate()
    vllm_proc.wait()
    print("✨ Standalone test complete.")