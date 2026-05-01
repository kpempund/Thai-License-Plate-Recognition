# Thai License Plate OCR & Post-Processing

Final stage of the pipeline: Extracts, sanitizes, and validates license plate text from cropped images.

| Item | Description |
|------|-------------|
| `ocr_pipeline.py` | The core engine: includes image upscaling, spatial routing, regex-based grammar enforcement, and fuzzy province matching. |
| `evaluation.py` | Benchmarking script that calculates Character Accuracy Rate (CAR) and Word Accuracy Rate (WAR). |
| `format_manual_datasets.py` | Preparation script: Decodes Thai character classes (TH01-TH37) and maps province codes to full Thai names to generate the labels.csv ground truth file. |
| `requirements.txt` | Python dependencies including `easyocr`, `opencv-python`, and `numpy`. |

## Pipeline Overview

| Step | Technique | What it does |
|------|-----------|--------------|
| 1 | **Dataset Formatting** | Processes raw CSV exports from Roboflow/TensorFlow, decodes the character map, and organizes images into the test_images/ folder for evaluation. |
| 2 | **Image Upscaling** | Resizes input by 200% and applies sharpening to fix blurry/low-res characters. |
| 3 | **Detection** | Runs EasyOCR with a strict Thai/Numeric `ALLOWLIST` to prevent hallucinations. |
| 4 | **Spatial Routing** | Uses Y-axis geometry (60/40 split) to separate the **Plate Number** from the **Province**. |
| 5 | **Grammar Sanitization** | Uses Regex to enforce `[Prefix][Consonants][Suffix]` rules and a dictionary to fix visual errors (e.g., `เ,โ,ไ` → `1`). |
| 6 | **Fuzzy Matching** | Uses Levenshtein distance to snap misspelled OCR text to the official list of 77 Thai provinces. |

## Results (Evaluation on 41 Images)

| Metric | Number Accuracy | Province Accuracy | Full Plate (Combined) |
|-------|--------|-----------|-----------|
| **CAR (Character)** | 76.51% | 67.04% | 68.40% |
| **WAR (Word/Full)** | 43.90% | 63.41% | **26.83%** |

*Note: WAR represents a perfect "Ground Truth" match. 26.83% represents plates where both the Number and Province were 100% correct.*

## Installation

1. Install PyTorch with CUDA support: [pytorch.org](https://pytorch.org/get-started/locally/)
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model & Logic Specs

- **Label Decoding:** Uses a custom CHAR_DECODER (TH01–TH37) and PROVINCE_MAP (e.g., BKK $\rightarrow$ กรุงเทพมหานคร) to translate technical dataset classes into human-readable Thai text.
- **Core Engine:** EasyOCR (CRAFT detector + CRNN recognizer)
- **Languages:** Thai (`th`), English (`en`)
- **Spatial Threshold:** 60% Y-axis split for Number vs. Province
- **Fuzzy Logic:** `difflib.get_close_matches` with a `0.3` cutoff for Province validation

### Post-Processing Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `upscale_factor` | 2.0 | Bicubic interpolation multiplier |
| `y_split_ratio` | 0.6 | Separates top row from bottom row |
| `confidence_min`| 0.1 | Ignores low-confidence OCR "ghost" text |
| `fuzzy_cutoff` | 0.3 | Forgiveness level for province typos |
