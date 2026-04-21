import json
import os
from pathlib import Path
from PIL import Image

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# 1. Load Surya Models (This will trigger a download the first time)
print("Loading Surya models (this may take a moment)...")
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

def validate_ocr(ocr_text, ground_truth_dict):
    """Checks if the JSON values exist in the raw OCR text."""
    total_fields = len(ground_truth_dict)
    found_fields = 0
    missing_fields = {}

    for key, expected_value in ground_truth_dict.items():
        str_value = str(expected_value).strip() 
        if str_value in ocr_text:
            found_fields += 1
        else:
            missing_fields[key] = str_value

    accuracy = (found_fields / total_fields) * 100 if total_fields > 0 else 0
    return accuracy, missing_fields

# 2. Setup Directories
base_dir = Path("extracted_data")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# 3. Process the Images
for img_path in images_dir.glob("*.*"):
    if img_path.is_file():
        print(f"\nProcessing {img_path.name}...")
        
        json_path = labels_dir / f"{img_path.stem}.json"
        
        if not json_path.exists():
            print(f"Skipping: No ground truth JSON found at {json_path}")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        # Run Surya OCR (Modern API)
        img = Image.open(img_path).convert("RGB")
        predictions = recognition_predictor([img], det_predictor=detection_predictor)

        # Combine all extracted text lines into one big string
        full_ocr_text = "\n".join([line.text for line in predictions[0].text_lines])

        # Validate
        accuracy, missing = validate_ocr(full_ocr_text, ground_truth)
        
        print(f"Accuracy: {accuracy:.2f}%")
        if missing:
            print(f"Failed to extract the following values:")
            for k, v in missing.items():
                print(f"  - {k}: {v}")