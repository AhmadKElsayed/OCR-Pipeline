import json
import torch
import gc
import os
import time
import fitz
from pathlib import Path
from PIL import Image

# Imports
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

# --- CONFIGURATION ---
INPUT_FILE = "DASA-Statement-1.png" # Change this to a PDF or Image
OUTPUT_FILE = "battleground_comparison.md"
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()

print(f"🚀 Battleground starting on {device}...")

# ---------------------------------------------------------
# 0. INPUT PARSER (Handle PDFs vs Images)
# ---------------------------------------------------------
images = []
is_pdf = INPUT_FILE.lower().endswith(".pdf")

if is_pdf:
    print(f"📄 PDF detected. Rasterizing pages...")
    doc = fitz.open(INPUT_FILE)
    for page in doc:
        # DPI 200 is usually the sweet spot for OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
else:
    print(f"🖼️ Image detected.")
    images = [Image.open(INPUT_FILE).convert("RGB")]

print(f"✅ Loaded {len(images)} page(s) for visual inference.")

# ---------------------------------------------------------
# 1. SURYA & MARKER
# ---------------------------------------------------------
print("\n--- Phase 1: Surya & Marker ---")
foundation_predictor = FoundationPredictor(device=device)
det_predictor = DetectionPredictor(device=device)
rec_predictor = RecognitionPredictor(foundation_predictor)

try:
    print("📝 Running Surya...")
    surya_preds = rec_predictor(images, det_predictor=det_predictor)
    surya_text = "\n\n".join(["\n".join([line.text for line in page_pred.text_lines]) for page_pred in surya_preds])
except Exception as e:
    surya_text = f"Surya Error: {e}"

print("📝 Running Marker...")
model_dict = create_model_dict()
marker_config = {"output_format": "markdown", "languages": "ar,en", "force_ocr": True, "device": device}
marker_converter = PdfConverter(config=marker_config, artifact_dict=model_dict)
marker_rendered = marker_converter(INPUT_FILE) # Marker handles both PDF and images natively
marker_text, _, _ = text_from_rendered(marker_rendered)

# 🔥 CRITICAL: Wipe Surya/Marker from VRAM
del rec_predictor, det_predictor, foundation_predictor, marker_converter, model_dict
clear_vram()

# ---------------------------------------------------------
# 2. DOCLING (RapidOCR + Table Structure)
# ---------------------------------------------------------
print("\n--- Phase 2: Docling ---")
ocr_options = RapidOcrOptions(force_full_page_ocr=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
pipeline_options.ocr_options = ocr_options

docling_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

try:
    docling_result = docling_converter.convert(INPUT_FILE)
    docling_text = docling_result.document.export_to_markdown()
except Exception as e:
    docling_text = f"Docling Error: {e}"

# 🔥 CRITICAL: Wipe Docling from VRAM
del docling_converter
clear_vram()

# ---------------------------------------------------------
# 3. QWEN 2.5-VL
# ---------------------------------------------------------
print("\n--- Phase 3: Qwen2.5-VL (Full Precision) ---")
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    dtype=torch.bfloat16,
    device_map="auto"
)
vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

vlm_prompt = """
Analyze this document carefully. 
1. Identify document type.
2. Extract key-value pairs and tables into JSON.
3. Transcribe handwriting.
4. Convert Eastern Arabic numerals (٤, ٥) to Western digits (4, 5).
Return the full text in the document.
"""

# Construct multipage vision prompt dynamically
qwen_content = [{"type": "image", "image": img, "max_pixels": 1500000} for img in images]
qwen_content.append({"type": "text", "text": vlm_prompt})

messages = [{"role": "user", "content": qwen_content}]
vlm_input_text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = vlm_processor(text=[vlm_input_text], images=image_inputs, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=1500)
    vlm_output = vlm_processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

# ---------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"# ⚔️ OCR & VLM Battleground\n**Source File:** `{INPUT_FILE}`\n\n")
    f.write("## 🟢 Surya Output\n```text\n" + surya_text + "\n```\n\n")
    f.write("## 🔵 Marker Output\n```markdown\n" + marker_text + "\n```\n\n")
    f.write("## 🔴 Docling Output\n```markdown\n" + docling_text + "\n```\n\n")
    f.write("## 🟡 Qwen2.5-VL Output\n" + vlm_output + "\n")

print(f"\n✨ Battle complete! Check {OUTPUT_FILE}")