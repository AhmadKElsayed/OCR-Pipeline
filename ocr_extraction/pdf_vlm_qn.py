import json
import re
import torch
import gc
from pathlib import Path
import pypdfium2 as pdfium
from qwen_vl_utils import process_vision_info
from thefuzz import fuzz 

# Imports for Qwen 2.5 VL and Quantization
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# 1. Hardware Setup & Model Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

print("⏳ Loading Qwen2.5-VL-7B in 4-bit Quantization (Fast & VRAM Efficient)...")

# --- NEW: Quantization Configuration ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Optimal 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16 # Keeps calculations fast and precise
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    quantization_config=quant_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

def validate_json_extraction(extracted_dict, ground_truth_dict):
    """
    Validates VLM output. Normalizes numbers (removes commas) and handles 
    RTL phone number flips automatically using token_set_ratio.
    """
    total_fields = len(ground_truth_dict)
    found_fields = 0
    results_detail = {}

    for key, expected_value in ground_truth_dict.items():
        val = str(expected_value).strip()
        
        if key not in extracted_dict:
            results_detail[key] = {"status": "FAIL (Missing Key)", "value": val}
            continue
            
        extracted_val = str(extracted_dict[key]).strip()
        
        # NORMALIZATION: Remove commas for numbers, standardize spaces
        clean_expected = val.replace(",", "").replace("+", " ").replace("-", " ")
        clean_extracted = extracted_val.replace(",", "").replace("+", " ").replace("-", " ")

        # 1. Direct Match (after removing commas)
        if clean_expected == clean_extracted:
            found_fields += 1
            results_detail[key] = {"status": "MATCH", "value": val}
        else:
            # 2. RTL Phone Number Handling
            if "phone" in key and fuzz.token_set_ratio(clean_expected, clean_extracted) == 100:
                found_fields += 1
                results_detail[key] = {"status": "MATCH", "value": val}
            else:
                # 3. True Mismatch
                results_detail[key] = {"status": f"FAIL (Extracted: {extracted_val})", "value": val}

    accuracy = (found_fields / total_fields) * 100 if total_fields > 0 else 0
    return accuracy, results_detail

# 2. Setup Directories
base_dir = Path("extracted_data")
pdfs_dir = base_dir / "pdfs"
labels_dir = base_dir / "labels"
results_dir = Path("vlm_qn_results")
results_dir.mkdir(exist_ok=True)

# 3. Process the PDFs
for pdf_path in pdfs_dir.glob("*.pdf"):
    print(f"\n📄 Processing {pdf_path.name} with Qwen2.5-VL (4-bit)...")
    
    json_path = labels_dir / f"{pdf_path.stem}.json"
    if not json_path.exists():
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    pdf_doc = pdfium.PdfDocument(pdf_path)
    page = pdf_doc[0]
    bitmap = page.render(scale=500/72).to_pil()
    image = bitmap.convert("RGB")

    # 4. Prepare the STRICT VLM Prompt
    json_keys = ", ".join(ground_truth.keys())
    prompt = f"""
    You are a STRICT document transcription engine. Your job is to extract data EXACTLY as it appears visually on the page.

    CRITICAL RULES:
    1. EXACT MATCH: Do NOT translate words. Do NOT use synonyms.
    2. THE SALARY TABLE: You MUST extract every single row. Look for these specific visual anchors:
       - For "basic": Find the number next to "Basic Salary" / "الراتب الأساسي".
       - For "housing": Find the number next to "Housing Allowance" / "بدل السكن".
       - For "transport": Find the number next to "Transport Allowance" / "بدل المواصلات".
       - For "other": Find the number next to "Other Allowances" / "بدلات أخرى".
       - For "total": Find the number next to "TOTAL MONTHLY SALARY" / "إجمالي الراتب الشهري".
       - For "words_total": Extract the full text next to "Total in Words" / "الإجمالي كتابةً".
    3. MISSING KEYS: You MUST include every single key from the list below in your final JSON, even if you think it is empty. Do not omit any keys.
    4. NO EXTRA TEXT: Return ONLY a valid JSON object.

    Use exclusively these exact keys: {json_keys}.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": 1500000},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 5. Generate Output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        clean_json_string = re.search(r'\{.*\}', output_text.replace('\n', ''), re.DOTALL).group()
        extracted_data = json.loads(clean_json_string)
    except Exception as e:
        print(f"⚠️ Failed to parse JSON from output: {e}")
        extracted_data = {}

    # 6. Validate
    accuracy, details = validate_json_extraction(extracted_data, ground_truth)

    # 7. Generate Output Formatted Document
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
        
        out.write("## 📝 Raw VLM Output\n")
        out.write("```json\n")
        out.write(json.dumps(extracted_data, indent=2, ensure_ascii=False))
        out.write("\n```\n")

    print(f"✅ Saved comparison to: {output_filename}")
    
    # --- MEMORY RESET ---
    del inputs, generated_ids, text, messages, image, pdf_doc
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n✨ Processing complete. Review files in '{results_dir}/'")