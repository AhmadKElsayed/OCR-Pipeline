import json
import os
import torch
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from thefuzz import fuzz

# Modern Surya Imports
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# 0. Hardware Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 1. Load Surya Models
print("Loading Surya models...")
foundation_predictor = FoundationPredictor(device=device)
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor(device=device)

def validate_ocr(ocr_text, ground_truth_dict, fuzzy_threshold=90):
    """
    Validates OCR text against ground truth with normalization and fuzzy matching.
    """
    # Normalize OCR: remove commas from numbers and handle common whitespace issues
    normalized_ocr = ocr_text.replace(",", "").strip()
    
    total_fields = len(ground_truth_dict)
    found_fields = 0
    results_detail = {}

    for key, expected_value in ground_truth_dict.items():
        val = str(expected_value).strip()
        
        # 1. Strict Check (with normalized commas)
        if val in normalized_ocr:
            found_fields += 1
            results_detail[key] = {"status": "MATCH", "value": val}
        
        # 2. Fuzzy Check (handles bidi-flipped phone numbers and minor typos)
        else:
            score = fuzz.partial_ratio(val, normalized_ocr)
            if score >= fuzzy_threshold:
                found_fields += 1
                results_detail[key] = {"status": f"FUZZY_MATCH ({score}%)", "value": val}
            else:
                results_detail[key] = {"status": "FAIL", "value": val, "score": score}

    accuracy = (found_fields / total_fields) * 100 if total_fields > 0 else 0
    return accuracy, results_detail

# 2. Setup Directories
base_dir = Path("extracted_data")
pdfs_dir = base_dir / "pdfs"
labels_dir = base_dir / "labels"
results_dir = Path("surya_results")
results_dir.mkdir(exist_ok=True)

# 3. Process the PDFs
for pdf_path in pdfs_dir.glob("*.pdf"):
    print(f"📄 Processing {pdf_path.name}...")
    
    json_path = labels_dir / f"{pdf_path.stem}.json"
    if not json_path.exists():
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Convert PDF to Images - Increased scale to 400 for smaller header text
    pdf_doc = pdfium.PdfDocument(pdf_path)
    full_ocr_text = ""
    
    for page_idx in range(len(pdf_doc)):
        page = pdf_doc[page_idx]
        # scale=400/72 provides higher resolution for small fonts
        bitmap = page.render(scale=400/72).to_pil()
        image = bitmap.convert("RGB")
        
        # Run Surya
        predictions = recognition_predictor([image], det_predictor=detection_predictor)
        page_text = "\n".join([line.text for line in predictions[0].text_lines])
        full_ocr_text += page_text + "\n"

    # Validate
    accuracy, details = validate_ocr(full_ocr_text, ground_truth)

    # 4. Generate the Comparison File
    output_filename = results_dir / f"{pdf_path.stem}_comparison.md"
    
    with open(output_filename, "w", encoding="utf-8") as out:
        out.write(f"# Comparison Results: {pdf_path.name}\n")
        out.write(f"**Calculated Accuracy:** {accuracy:.2f}%\n\n")
        
        out.write("## 📝 Field Validation Details\n")
        out.write("| Field | Status | Expected Value |\n")
        out.write("| :--- | :--- | :--- |\n")
        for k, v in details.items():
            status_icon = "✅" if "MATCH" in v["status"] else "❌"
            out.write(f"| **{k}** | {status_icon} {v['status']} | `{v['value']}` |\n")
        
        out.write("\n---\n\n")
        out.write("## 🎯 Full Ground Truth JSON\n")
        out.write("```json\n")
        out.write(json.dumps(ground_truth, indent=2, ensure_ascii=False))
        out.write("\n```\n\n")
        
        out.write("## 📝 Raw Surya OCR Output\n")
        out.write("```text\n")
        out.write(full_ocr_text.strip())
        out.write("\n```\n")

    print(f"✅ Saved comparison to: {output_filename}")

print("\n✨ Processing complete. Review files in 'comparison_results/'")