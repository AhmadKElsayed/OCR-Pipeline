import json
import os
from pathlib import Path

# 1. Define the directory and the keys to remove
labels_dir = Path("extracted_data/labels")
keys_to_remove = [
    "company_name_en",
    "company_address_en",
    "company_email",
    "employee_name_en"
]

def clean_json_files():
    if not labels_dir.exists():
        print(f"❌ Error: Directory {labels_dir} not found.")
        return

    json_files = list(labels_dir.glob("*.json"))
    print(f"🔍 Found {len(json_files)} JSON files. Starting cleanup...")

    count = 0
    for json_path in json_files:
        try:
            # Read the current content
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Remove the problematic keys
            updated = False
            for key in keys_to_remove:
                if key in data:
                    del data[key]
                    updated = True

            # Save the file back only if changes were made
            if updated:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                count += 1
        
        except Exception as e:
            print(f"⚠️ Failed to process {json_path.name}: {e}")

    print(f"✨ Cleanup complete! Updated {count} files.")

if __name__ == "__main__":
    clean_json_files()