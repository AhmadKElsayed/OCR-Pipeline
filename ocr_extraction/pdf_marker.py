import json
import os
import torch
from pathlib import Path
from thefuzz import fuzz

# Marker Imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# 0. Hardware & Model Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

print("Loading Marker models (Layout + Surya)...")
model_dict = create_model_dict() 

# 1. Fixed Configuration
# We pass the dict directly to the converter to fix the ValueError
config = {
    "output_format": "markdown",
    "languages": "ar", 
    "force_ocr": True,
    "device": device
}

converter = PdfConverter(
    config=config, 
    artifact_dict=model_dict
)

def validate_extraction(extracted_text, ground_truth_dict, fuzzy_threshold=90):
    """
    Validates extracted text against ground truth using fuzzy matching.
    """
    # Standardize spaces and remove commas for numerical comparison
    clean_text = " ".join(extracted_text.replace(",", "").split())
    
    total_fields = len(ground_truth_dict)
    found_fields = 0
    results_detail = {}

    for key, expected_value in ground_truth_dict.items():
        val = str(expected_value).strip()
        
        # token_set_ratio is best for Marker as it handles layout shifts
        score = fuzz.token_set_ratio(val, clean_text)
        
        if score >= fuzzy_threshold:
            found_fields += 1
            results_detail[key] = {"status": "MATCH", "value": val}
        else:
            results_detail[key] = {"status": f"FAIL ({score}%)", "value": val}

    accuracy = (found_fields / total_fields) * 100 if total_fields > 0 else 0
    return accuracy, results_detail

# 2. Setup Directories
base_dir = Path("extracted_data")
pdfs_dir = base_dir / "pdfs"
labels_dir = base_dir / "labels"
results_dir = Path("marker_results")
results_dir.mkdir(exist_ok=True)

# 3. Process the PDFs
for pdf_path in pdfs_dir.glob("*.pdf"):
    print(f"📄 Processing {pdf_path.name} with Marker...")
    
    json_path = labels_dir / f"{pdf_path.stem}.json"
    if not json_path.exists():
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    try:
        # Run conversion
        rendered = converter(str(pdf_path))
        # Extract text content
        full_text, _, _ = text_from_rendered(rendered)
    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        continue

    # Validate results
    accuracy, details = validate_extraction(full_text, ground_truth)

    # 4. Generate Report
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
        
        out.write("## 📝 Raw Marker Output (Markdown)\n")
        out.write("```markdown\n")
        out.write(full_text.strip())
        out.write("\n```\n")

    print(f"✅ Saved comparison to: {output_filename}")

print(f"\n✨ Processing complete. Review files in '{results_dir}/'")