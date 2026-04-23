import json
import torch
import gc
import os
import time
import io
import base64
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from openai import OpenAI
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
device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()

def encode_image(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def run_battleground(test_dir: str, output_dir: str = "battleground_results"):
    """
    Batch-Optimized Pipeline: 
    Loads each model ONCE, processes all files, and unloads it before loading the next model.
    This entirely prevents VRAM fragmentation and crashes.
    """
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
    
    if not test_path.exists():
        print(f"❌ Error: The directory '{test_dir}' does not exist.")
        return
        
    files = sorted([f for f in test_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    
    if not files:
        print(f"⚠️ No valid documents found in '{test_dir}'.")
        return

    print(f"🚀 Found {len(files)} files. Initializing TRUE Batch-Optimized Run...")

    # Master dictionary to hold all results in memory until the final audit
    batch_data = {}

    # ---------------------------------------------------------
    # PHASE 0: PRE-PROCESS ALL IMAGES INTO RAM
    # ---------------------------------------------------------
    print("\n--- Phase 0: Pre-Processing Documents ---")
    for file_path in files:
        input_str = str(file_path)
        file_ext = file_path.suffix.lower()
        images = []
        try:
            if file_ext == ".pdf":
                doc = fitz.open(input_str)
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
                    images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                doc.close()
            else:
                images = [Image.open(input_str).convert("RGB")]
            
            batch_data[input_str] = {
                "file_path": file_path,
                "images": images,
                "judge_img": images[0],
                "reports": {}
            }
        except Exception as e:
            print(f"❌ Failed to parse {file_path.name}: {e}")

    active_files = [f for f in files if str(f) in batch_data]

    # ---------------------------------------------------------
    # PHASE 1: SURYA & MARKER
    # ---------------------------------------------------------
    print("\n" + "="*60 + "\n--- Phase 1: Surya & Marker ---\n" + "="*60)
    try:
        print("⏳ Loading Surya & Marker (Just once!)...")
        f_pred = FoundationPredictor(device=device)
        d_pred = DetectionPredictor(device=device)
        r_pred = RecognitionPredictor(f_pred)
        
        m_dict = create_model_dict()
        m_conv = PdfConverter(config={"output_format": "markdown", "languages": "ar,en", "force_ocr": True, "device": device}, artifact_dict=m_dict)

        for file_path in active_files:
            input_str = str(file_path)
            print(f"  -> Surya & Marker extracting: {file_path.name} ...")
            
            # Surya
            s_start = time.time()
            try:
                surya_text = "\n\n".join(["\n".join([l.text for l in p.text_lines]) for p in r_pred(batch_data[input_str]["images"], det_predictor=d_pred)])
                batch_data[input_str]["reports"]["Surya"] = {"text": surya_text, "time": time.time() - s_start}
            except Exception as e:
                batch_data[input_str]["reports"]["Surya"] = {"text": f"Error: {e}", "time": 0}

            # Marker
            m_start = time.time()
            try:
                m_rendered = m_conv(input_str)
                marker_text, _, _ = text_from_rendered(m_rendered)
                batch_data[input_str]["reports"]["Marker"] = {"text": marker_text, "time": time.time() - m_start}
            except Exception as e:
                batch_data[input_str]["reports"]["Marker"] = {"text": f"Error: {e}", "time": 0}

        print("🛑 Unloading Surya & Marker forever...")
        del r_pred, d_pred, f_pred, m_conv, m_dict
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
    clear_vram()

    # ---------------------------------------------------------
    # PHASE 2: DOCLING
    # ---------------------------------------------------------
    print("\n" + "="*60 + "\n--- Phase 2: Docling ---\n" + "="*60)
    try:
        print("⏳ Loading Docling models...")
        ocr_options = RapidOcrOptions(force_full_page_ocr=True)
        pipeline_options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
        pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
        pipeline_options.ocr_options = ocr_options
        d_conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

        for file_path in active_files:
            input_str = str(file_path)
            print(f"  -> Docling extracting: {file_path.name} ...")
            start_time = time.time()
            try:
                docling_text = d_conv.convert(input_str).document.export_to_markdown()
                batch_data[input_str]["reports"]["Docling"] = {"text": docling_text, "time": time.time() - start_time}
            except Exception as e:
                batch_data[input_str]["reports"]["Docling"] = {"text": f"Error: {e}", "time": 0}
        
        print("🛑 Unloading Docling models from RAM...")
        del d_conv
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
    clear_vram()

    # ---------------------------------------------------------
    # PHASE 3: QWEN 2.5-VL
    # ---------------------------------------------------------
    print("\n" + "="*60 + "\n--- Phase 3: Qwen2.5-VL ---\n" + "="*60)
    try:
        print("⏳ Loading Qwen (Just once!)...")
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device_map=device)
        vlm_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        vlm_prompt = """
            Perform pure OCR on this document.
            Extract all text and tables and format the output in clean Markdown. 
            Preserve table structures exactly as they appear.
            Transcribe any handwriting.
            Convert Eastern Arabic numerals (٤, ٥) to Western digits (4, 5).
            Do NOT output JSON or key-value pairs. Output only the transcribed Markdown text.
            """

        for file_path in active_files:
            input_str = str(file_path)
            print(f"  -> Qwen extracting: {file_path.name} ...")
            start_time = time.time()
            try:
                # Qwen natively protects against massive tokens via max_pixels, no thumbnailing needed here
                q_msg = [{"role": "user", "content": [{"type": "image", "image": img, "max_pixels": 1500000} for img in batch_data[input_str]["images"]] + [{"type": "text", "text": vlm_prompt}]}]
                vlm_in = vlm_proc.apply_chat_template(q_msg, tokenize=False, add_generation_prompt=True)
                vis_in, _ = process_vision_info(q_msg)
                inputs = vlm_proc(text=[vlm_in], images=vis_in, padding=True, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    ids = vlm_model.generate(**inputs, max_new_tokens=1500)
                    qwen_text = vlm_proc.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                
                batch_data[input_str]["reports"]["Qwen2.5-VL"] = {"text": qwen_text, "time": time.time() - start_time}
            except Exception as e:
                batch_data[input_str]["reports"]["Qwen2.5-VL"] = {"text": f"Error: {e}", "time": 0}
                
        print("🛑 Unloading Qwen from GPU forever...")
        del vlm_model, vlm_proc
    except Exception as e:
        for file_path in active_files:
            batch_data[str(file_path)]["reports"]["Qwen2.5-VL"] = {"text": f"Error: {e}", "time": 0}
    clear_vram()


    # ---------------------------------------------------------
    # PHASE 4: GRAND AUDIT & EXPORT
    # ---------------------------------------------------------
    print("\n" + "="*60 + "\n--- Phase 4: The Grand Audit (GPT-4o-mini) ---\n" + "="*60)
    
    for file_path in active_files:
        input_str = str(file_path)
        print(f"⚖️ Auditing {file_path.name} with OpenAI...")
        report_data = batch_data[input_str]["reports"]
        judge_img = batch_data[input_str]["judge_img"]

        audit_prompt = f"""
        Act as a professional Bilingual OCR (Optical Character Recognition) Auditor. Compare the attached original document image with the outputs from {len(report_data)} different OCR methods.
        
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
        Output ONLY a JSON dictionary where the keys are the exact method names provided above:
        {
          "MethodName1": {"score": 1-10, "feedback": "short critique", "arabic_quality": "excellent/fair/poor", "english_quality": "excellent/fair/poor"},
          "MethodName2": {"score": 1-10, "feedback": "short critique", "arabic_quality": "excellent/fair/poor", "english_quality": "excellent/fair/poor"}
        }"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": audit_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(judge_img)}"}}]
                }], response_format={"type": "json_object"}
            )
            audit_results = json.loads(response.choices[0].message.content)
            for method in report_data.keys():
                report_data[method]["judge"] = audit_results.get(method, {"score": 0, "feedback": "Judge missed this.", "arabic_quality": "N/A", "english_quality": "N/A"})
        except Exception as e:
            for method in report_data.keys():
                report_data[method]["judge"] = {"score": 0, "feedback": "API Error", "arabic_quality": "N/A", "english_quality": "N/A"}

        # SAVE REPORT
        output_file = out_path / f"{file_path.stem}_report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# 🛡️ OCR Battleground & AI Audit\n\n**Target File:** `{file_path.name}`\n\n## 📊 Leaderboard\n")
            f.write("| Method | Score | Time (s) | Arabic Quality | English Quality | Feedback |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            sorted_methods = sorted(report_data.items(), key=lambda x: x[1].get("judge", {}).get("score", 0), reverse=True)
            for name, data in sorted_methods:
                j = data.get("judge", {})
                f.write(f"| **{name}** | **{j.get('score', 0)}/10** | {data['time']:.2f}s | {j.get('arabic_quality', 'N/A').title()} | {j.get('english_quality', 'N/A').title()} | {j.get('feedback', 'Error')} |\n")
            
            f.write("\n---\n")
            for name, data in report_data.items():
                f.write(f"## 🛠️ {name} Raw Output\n```text\n{data['text']}\n```\n\n")

        print(f"✨ Saved {output_file.name}")

    print("\n🎉 ALL FILES PROCESSED SUCCESSFULLY!")