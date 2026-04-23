import json
import torch
import gc
import os
import time
import io
import base64
import requests
import subprocess
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from openai import OpenAI
import ollama
from dotenv import load_dotenv

# OCR & VLM Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableStructureOptions
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- SETUP & CONFIG ---
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "DASA-Statement-1.png" 
OUTPUT_FILE = "battleground_report.md"
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()

def encode_image(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ---------------------------------------------------------
# 0. INPUT PARSER (PDFs & Images)
# ---------------------------------------------------------
images = []
file_ext = Path(INPUT_FILE).suffix.lower()

if file_ext == ".pdf":
    print(f"📄 Processing PDF...")
    doc = fitz.open(INPUT_FILE)
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
elif file_ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
    print(f"🖼️ Processing Image...")
    images = [Image.open(INPUT_FILE).convert("RGB")]
else:
    raise ValueError(f"Unsupported file type: {file_ext}")

judge_img = images[0] # Image to send to OpenAI
report_data = {}

# ---------------------------------------------------------
# 1. SURYA & MARKER
# ---------------------------------------------------------
print("\n--- Phase 1: Surya & Marker ---")

print("📝 Running Surya...")
start_time = time.time()
try:
    f_pred = FoundationPredictor(device=device)
    d_pred = DetectionPredictor(device=device)
    r_pred = RecognitionPredictor(f_pred)
    surya_text = "\n\n".join(["\n".join([l.text for l in p.text_lines]) for p in r_pred(images, det_predictor=d_pred)])
    report_data["Surya"] = {"text": surya_text, "time": time.time() - start_time}
    del r_pred, d_pred, f_pred
except Exception as e:
    report_data["Surya"] = {"text": f"Error: {e}", "time": 0}
clear_vram()

print("📝 Running Marker...")
start_time = time.time()
try:
    m_dict = create_model_dict()
    m_conv = PdfConverter(config={"output_format": "markdown", "languages": "ar,en", "force_ocr": True, "device": device}, artifact_dict=m_dict)
    m_rendered = m_conv(INPUT_FILE)
    marker_text, _, _ = text_from_rendered(m_rendered)
    report_data["Marker"] = {"text": marker_text, "time": time.time() - start_time}
    del m_conv, m_dict
except Exception as e:
    report_data["Marker"] = {"text": f"Error: {e}", "time": 0}
clear_vram()

# ---------------------------------------------------------
# 2. DOCLING
# ---------------------------------------------------------
print("\n--- Phase 2: Docling ---")
start_time = time.time()
try:
    ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    pipeline_options.ocr_options = ocr_options

    d_conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    docling_text = d_conv.convert(INPUT_FILE).document.export_to_markdown()
    report_data["Docling"] = {"text": docling_text, "time": time.time() - start_time}
    del d_conv
except Exception as e:
    report_data["Docling"] = {"text": f"Error: {e}", "time": 0}
clear_vram()

# ---------------------------------------------------------
# 3. DOTS.MOCR + GEMMA (OLLAMA)
# ---------------------------------------------------------
print("\n--- Phase 3: dots.mocr + Gemma (Ollama) ---")
start_time = time.time()
vllm_proc = None

try:
    print("⏳ Spinning up vLLM server for dots.mocr...")
    vllm_proc = subprocess.Popen([
        "uv", "run", "vllm", "serve", "rednote-hilab/dots.mocr",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.6", 
        "--chat-template-content-format", "string",
        "--trust-remote-code", "--port", "8000", "--max-model-len", "8192"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    vllm_ready = False
    poll_start = time.time()
    
    while time.time() - poll_start < 120:
        # Check if the process died prematurely
        if vllm_proc.poll() is not None:
            crash_log = vllm_proc.stdout.read()
            raise Exception(f"vLLM process crashed instantly!\nLogs:\n{crash_log}")

        try:
            if requests.get("http://localhost:8000/health").status_code == 200:
                vllm_ready = True
                break
        except requests.exceptions.ConnectionError:
            time.sleep(2)
            print(".", end="", flush=True)

    if not vllm_ready:
        timeout_log = vllm_proc.stdout.read()
        raise Exception(f"vLLM timed out after 120 seconds.\nLogs:\n{timeout_log}")

    print("\n🟢 vLLM ready. Extracting raw text with dots.mocr...")
    client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    
    raw_ocr_combined = ""
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        
        resp = client.chat.completions.create(
            model="rednote-hilab/dots.mocr",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "text", "text": "<|img|><|imgpad|><|endofimg|>Extract the text content from this image."}
            ]}],
            max_completion_tokens=4096, temperature=0.1
        )
        raw_ocr_combined += resp.choices[0].message.content + "\n"

    print("🧠 Structuring text into JSON using Ollama (gemma3n:e2b)...")
    FINANCE_SYSTEM = """You are a financial extraction expert.
    Given raw OCR text from a bank statement, return ONLY a JSON object with this exact schema:
    {
      "summary": { "opening_balance": float, "closing_balance": float, "total_deposits": float, "total_withdrawals": float },
      "transactions": [ {"date": "string", "description": "string", "amount": float, "type": "debit or credit"} ]
    }
    Rules: use positive numbers for both debits and credits. Output raw JSON only."""

    msgs = [
        {"role": "system", "content": FINANCE_SYSTEM},
        {"role": "user", "content": f"Bank statement text:\n\n{raw_ocr_combined}"},
    ]
    ollama_resp = ollama.chat(model="gemma3n:e2b", messages=msgs, format="json", options={"temperature": 0})
    
    formatted_output = f"```json\n{ollama_resp['message']['content']}\n```"
    report_data["dots.mocr+Gemma"] = {"text": formatted_output, "time": time.time() - start_time}

except Exception as e:
    print(f"\n❌ Phase 3 Failed: {e}")
    report_data["dots.mocr+Gemma"] = {"text": f"Error: {e}", "time": 0}
finally:
    if vllm_proc is not None and vllm_proc.poll() is None:
        print("\n🛑 Shutting down vLLM server to free memory...")
        vllm_proc.terminate()
        vllm_proc.wait()
    clear_vram()
    time.sleep(3)


# ---------------------------------------------------------
# 4. QWEN 2.5-VL
# ---------------------------------------------------------
print("\n--- Phase 4: Qwen2.5-VL ---")
start_time = time.time()
try:
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device_map="auto")
    vlm_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    vlm_prompt = """
        Perform pure OCR on this document.
        Extract all text and tables and format the output in clean Markdown. 
        Preserve table structures exactly as they appear.
        Transcribe any handwriting.
        Convert Eastern Arabic numerals (٤, ٥) to Western digits (4, 5).
        Do NOT output JSON or key-value pairs. Output only the transcribed Markdown text.
        """
    q_msg = [{"role": "user", "content": [{"type": "image", "image": img, "max_pixels": 1500000} for img in images] + [{"type": "text", "text": vlm_prompt}]}]
    vlm_in = vlm_proc.apply_chat_template(q_msg, tokenize=False, add_generation_prompt=True)
    vis_in, _ = process_vision_info(q_msg)
    inputs = vlm_proc(text=[vlm_in], images=vis_in, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        ids = vlm_model.generate(**inputs, max_new_tokens=1500)
        qwen_text = vlm_proc.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    report_data["Qwen2.5-VL"] = {"text": qwen_text, "time": time.time() - start_time}
    del vlm_model, vlm_proc
except Exception as e:
    report_data["Qwen2.5-VL"] = {"text": f"Error: {e}", "time": 0}
clear_vram()


# ---------------------------------------------------------
# 5. THE GRAND AUDIT (OpenAI Consolidated Judge)
# ---------------------------------------------------------
print("\n--- Phase 5: The Grand Audit (GPT-4o) ---")
print("⚖️ Sending image and all outputs to OpenAI for judging...")

# Build a massive prompt containing all extractions cleanly separated
audit_prompt = f"""
Act as a professional Bilingual OCR (Optical Character Recognition) Auditor. 
Compare the attached original document image with the outputs from {len(report_data)} different OCR methods.

**CRITICAL INSTRUCTION:** This is a STRICT OCR evaluation. Do NOT evaluate or penalize the format of the output (e.g., Markdown vs. Plain Text). Ignore whether the data is structured into key-value pairs. Your ONLY job is to evaluate how perfectly the raw text and numbers from the image were transcribed.

Evaluate each method based ONLY on:
1. Numerical & Financial Accuracy: Did it perfectly capture all amounts, dates, balances, and account numbers without skipping digits or hallucinating?
2. Bilingual Text Accuracy: Did it accurately transcribe both the Arabic and English characters without missing words or jumbling the reading direction (Right-to-Left vs Left-to-Right)?
3. Table Reading Order: Did it read the table cells in the correct logical sequence?

Here are the extractions:
"""

for method_name, data in report_data.items():
    audit_prompt += f"\n\n--- BEGIN {method_name} OUTPUT ---\n{data['text']}\n--- END {method_name} OUTPUT ---\n"

audit_prompt += """
Output ONLY a JSON dictionary where the keys are the exact method names provided above, and the values are their score sheets:
{
  "MethodName1": {
    "score": 1-10, 
    "feedback": "short critique on character/number accuracy and table reading order (do not mention JSON/formatting)", 
    "arabic_quality": "excellent/fair/poor",
    "english_quality": "excellent/fair/poor"
  },
  "MethodName2": {
    "score": 1-10, 
    "feedback": "short critique on character/number accuracy and table reading order (do not mention JSON/formatting)", 
    "arabic_quality": "excellent/fair/poor",
    "english_quality": "excellent/fair/poor"
  }
}
"""

try:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": audit_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(judge_img)}"}},
            ],
        }],
        response_format={"type": "json_object"}
    )
    audit_results = json.loads(response.choices[0].message.content)
    
    # Merge the judge's scores back into our report dictionary
    for method in report_data.keys():
        report_data[method]["judge"] = audit_results.get(method, {
            "score": 0, 
            "feedback": "Judge missed this.", 
            "arabic_quality": "N/A",
            "english_quality": "N/A"
        })
        
except Exception as e:
    print(f"❌ OpenAI Judge Failed: {e}")
    for method in report_data.keys():
        report_data[method]["judge"] = {
            "score": 0, 
            "feedback": "API Error", 
            "arabic_quality": "N/A",
            "english_quality": "N/A"
        }

# ---------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"# 🛡️ OCR Battleground & AI Audit\n\n")
    f.write(f"**Target File:** `{INPUT_FILE}`  \n")
    f.write(f"**Total Pages:** {len(images)}  \n\n")
    
    f.write("## 📊 Leaderboard\n")
    f.write("| Method | Score | Time (s) | Arabic Quality | English Quality | Feedback |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    
    # Sort the report data by score (highest first)
    sorted_methods = sorted(report_data.items(), key=lambda x: x[1].get("judge", {}).get("score", 0), reverse=True)
    
    for name, data in sorted_methods:
        j = data.get("judge", {})
        f.write(f"| **{name}** | **{j.get('score', 0)}/10** | {data['time']:.2f}s | {j.get('arabic_quality', 'N/A').title()} | {j.get('english_quality', 'N/A').title()} | {j.get('feedback', 'Error')} |\n")
    
    f.write("\n---\n")
    for name, data in report_data.items():
        f.write(f"## 🛠️ {name} Raw Output\n```text\n{data['text']}\n```\n\n")

print(f"\n✨ Battle complete! Check {OUTPUT_FILE}")