# Multilingual OCR & VLM Document Extraction Battleground

This repository contains a robust, multi-model evaluation pipeline designed to extract structured data (JSON/Markdown) from complex, mixed-script documents (Arabic/English). It is specifically optimized to handle challenging layouts, tables, and handwritten Eastern Arabic numerals commonly found in medical receipts and financial certificates.

## 🚀 Models & Frameworks Integrated

This pipeline pits four distinct approaches against each other:
1. **Surya OCR:** Raw character recognition and detection (highly accurate for Arabic script).
2. **Marker:** Layout-aware document parsing that reconstructs tables into Markdown (powered by Surya).
3. **Docling:** Document parsing configured specifically with `RapidOcrOptions` (PaddleOCR) for optimized bi-directional text support.
4. **Qwen2.5-VL (7B):** A state-of-the-art Vision-Language Model loaded in native `bfloat16` precision for deep visual reasoning, handwriting transcription, and strict JSON schema adherence.

## 🧠 Architecture & VRAM Management

Running multiple state-of-the-art vision models simultaneously requires careful memory orchestration. This pipeline uses a **Sequential Loading Architecture** to prevent `CUDA OutOfMemory` errors on 24GB GPUs (like the NVIDIA L4).

The script executes in phases, actively wiping memory (`torch.cuda.empty_cache()` and `gc.collect()`) between the layout engines (Marker/Docling) and the heavy Vision-Language Model (Qwen2.5-VL).

## 🛠️ Installation & Setup

### Prerequisites
* **OS:** Linux (Ubuntu recommended)
* **GPU:** NVIDIA GPU with at least 24GB VRAM (e.g., L4, RTX 3090/4090)
* **Python:** 3.10 or 3.11 (Note: `surya-ocr` multiprocessing can encounter instability on Python 3.13)
* **Package Manager:** `uv` is highly recommended for fast dependency resolution.

### Install Dependencies
Create your virtual environment and install the required libraries:

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision accelerate bitsandbytes
uv pip install surya-ocr marker-pdf docling transformers qwen-vl-utils thefuzz
```

## 📂 Project Structure

```text
├── extracted_data/
│   ├── pdfs/           # Place your raw PDF files here
│   └── labels/         # Place ground-truth JSON schemas here
├── full_test.py        # The main sequential testing script
├── .gitignore
└── README.md
```

## ⚡ Usage

To run the full comparison on a target image or document:

1. Place your target image (e.g., `document.jpg`) in the root directory or update the `image_path` in the script.
2. Execute the battleground script:
```bash
python full_test.py
```

### Outputs
The script will sequentially load the models, process the document, and generate a `battleground_comparison.md` file. This file contains a side-by-side comparison of:
* Surya's raw text detection
* Marker's Markdown reconstruction
* Docling's Layout parsing
* Qwen2.5-VL's structured JSON output

## 🧪 Validation & Fuzzy Matching
The pipeline includes a custom `validate_extraction` function utilizing `thefuzz`. This allows for intelligent accuracy scoring against ground-truth JSONs by:
* Normalizing commas and spaces in numerical data.
* Applying fuzzy logic to handle RTL (Right-to-Left) phone number segments that occasionally flip during layout reconstruction.
* Aggressively normalizing Arabic spacing for strict character-matching accuracy.