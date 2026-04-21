import json
from pathlib import Path
import re
from thefuzz import fuzz

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat

print("Configuring Docling with RapidOCR for Arabic support...")

# 1. Initialize RapidOCR (PaddleOCR under the hood)
ocr_options = RapidOcrOptions()

# 2. Attach OCR to the PDF Pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = ocr_options

# 3. Initialize Converter with the custom pipeline
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def validate_conversion(extracted_text, ground_truth_dict, fuzzy_threshold=85):
    """
    Validates text with aggressive Arabic RTL/Space normalization.
    """
    clean_text = extracted_text.replace(",", "")
    clean_text = " ".join(clean_text.split())
    clean_text = clean_text.replace("+", " ").replace("-", " ")
    
    total_fields = len(ground_truth_dict)
    found_fields = 0
    results_detail = {}

    for key, expected_value in ground_truth_dict.items():
        val = str(expected_value).strip()
        clean_val = val.replace("+", " ").replace("-", " ")
        
        # Arabic Space Normalization: If the key ends in _ar, we remove spaces 
        # from both the expected value and the OCR text for a pure character check.
        if key.endswith("_ar"):
            pure_ar_expected = clean_val.replace(" ", "")
            pure_ar_ocr = clean_text.replace(" ", "")
            
            # Use strict inclusion or very high fuzzy match for pure strings
            if pure_ar_expected in pure_ar_ocr:
                found_fields += 1
                results_detail[key] = {"status": "MATCH", "value": val}
            else:
                score = fuzz.partial_ratio(pure_ar_expected, pure_ar_ocr)
                if score >= fuzzy_threshold:
                    found_fields += 1
                    results_detail[key] = {"status": f"FUZZY_MATCH ({score}%)", "value": val}
                else:
                    results_detail[key] = {"status": f"FAIL (Best match: {score}%)", "value": val}
        else:
            # Standard English/Number Check
            if clean_val in clean_text:
                found_fields += 1
                results_detail[key] = {"status": "MATCH", "value": val}
            else:
                score = fuzz.token_set_ratio(clean_val, clean_text)
                if score >= fuzzy_threshold:
                    found_fields += 1
                    results_detail[key] = {"status": f"FUZZY_MATCH ({score}%)", "value": val}
                else:
                    results_detail[key] = {"status": f"FAIL (Best match: {score}%)", "value": val}

    accuracy = (found_fields / total_fields) * 100 if total_fields > 0 else 0
    return accuracy, results_detail

# Setup Directories
base_dir = Path("extracted_data")
pdfs_dir = base_dir / "pdfs"
labels_dir = base_dir / "labels"
results_dir = Path("docling_rapid_results")
results_dir.mkdir(exist_ok=True)

# Process the PDFs
for pdf_path in pdfs_dir.glob("*.pdf"):
    print(f"📄 Processing {pdf_path.name}...")
    
    json_path = labels_dir / f"{pdf_path.stem}.json"
    if not json_path.exists():
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    # Run Conversion
    result = converter.convert(pdf_path)
    md_output = result.document.export_to_markdown()

    # Validate
    accuracy, details = validate_conversion(md_output, ground_truth)

    # Generate Report
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

        out.write("## 📝 Raw Docling Markdown Output\n")
        out.write("```markdown\n")
        out.write(md_output.strip())
        out.write("\n```\n")

    print(f"✅ Saved comparison to: {output_filename}")

print(f"\n✨ Processing complete. Review files in '{results_dir}/'")