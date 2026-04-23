# 🛡️ Multilingual OCR & VLM Document Extraction Battleground

This repository contains a robust, multi-model evaluation pipeline designed to extract structured text and Markdown from complex, mixed-script documents (Arabic/English). It is specifically optimized to handle challenging layouts, financial tables, and handwritten Eastern Arabic numerals commonly found in medical receipts and government certificates.

## 🚀 Models & Frameworks Integrated

This pipeline pits five distinct approaches against each other:
1. **Qwen2.5-VL (7B):** A state-of-the-art multimodal model loaded natively via Transformers for deep visual reasoning and strict layout adherence.
2. **Surya OCR:** Raw character recognition and detection (highly accurate for dense Arabic script).
3. **Marker:** Layout-aware document parsing that reconstructs tables into Markdown (powered by Surya).
4. **Docling:** Document parsing configured specifically with `RapidOcrOptions` (PaddleOCR) for optimized bi-directional text support.

## ⚖️ The Grand Audit (GPT-4o)
Instead of relying purely on fuzzy string matching, this pipeline features an automated AI Judge. At the end of the pipeline, **GPT-4o-mini** is given the original document image and the raw extractions from all 5 models. 

It acts as a Bilingual OCR Auditor, scoring each model on a scale of 1-10 based strictly on:
* **Numerical & Financial Accuracy:** Zero-tolerance for hallucinated digits or missed decimal points.
* **Bilingual Text Accuracy:** Perfect transcription of Arabic/English scripts without jumbling Right-to-Left directions.
* **Table Reading Order:** Ensuring complex financial grids are read sequentially.

## 🧠 True Batch-Optimized Architecture

Running multiple 15-gigabyte state-of-the-art vision models sequentially on a single GPU usually results in severe `CUDA OutOfMemory` fragmentation. 

This pipeline uses a **Phase-by-Phase Batch Architecture**. Instead of loading and unloading models for every single file, it loads a model (e.g., Qwen) *once*, processes the entire directory of documents into memory, and completely eradicates the model from the GPU before spinning up the next one (e.g., Docling). This guarantees zero VRAM crashing on 24GB GPUs (like the NVIDIA L4).

## 🛠️ Installation & Setup

### Prerequisites
* **OS:** Linux (Ubuntu recommended)
* **GPU:** NVIDIA GPU with at least 24GB VRAM (e.g., L4, RTX 3090/4090)
* **Python:** 3.10 or 3.11 
* **Package Manager:** `uv` is highly recommended for fast dependency resolution.
* **Ollama:** Required for the Phase 1 Markdown structuring (Gemma 3).

### Environment Variables
Create a `.env` file in the root directory and add your OpenAI key for the Grand Audit:
```env
OPENAI_API_KEY=sk-proj-...
```

### Install Dependencies
Create your virtual environment and install the required libraries:

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision accelerate bitsandbytes
uv pip install surya-ocr marker-pdf docling transformers qwen-vl-utils vllm openai python-dotenv ollama
```

### Initialize Ollama
You must start the Ollama server and pull the structuring model before running Phase 1:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > ollama.log 2>&1 &
ollama pull gemma3n:e2b
```

## 📂 Project Structure

```text
├── test_data/              # Place your raw PDFs and Images here
├── battleground_results/   # Auto-generated markdown leaderboards
├── ocr_evaluator.py        # The core pipeline logic
├── run_tests.py            # The simple execution script
├── .gitignore
└── README.md
```

## ⚡ Usage

To run the full batch comparison on a directory of images or PDFs:

1. Place your target documents in the `test_data/` folder.
2. Execute the battleground script:
```bash
python run_tests.py
```

### Outputs
The script will sequentially load the models, process the entire directory, and generate a dedicated `[filename]_report.md` for each document in the `battleground_results/` folder. 

Each report contains:
1. **The Leaderboard:** A sorted Markdown table showing the GPT-4o Audit Score, processing time, and specific feedback for each model.
2. **Raw Extractions:** The raw output from all 5 models attached below for manual review.