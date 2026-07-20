"""
======================================
=== Generate Card Text OCR Dataset ===
======================================

Use IBM docling to iterate through the collection of magic cards for which we
have scryfall tags extracted to generate raw text. This will be used downstream
to test a variant of the scryfall autotagger model which relies on card images
instead of on highly structured Scryfall-sourced text.

NOTE: Base Docling takes ~3-5 seconds per card.

Pre Run Checklist:
1. Verify that data/card_image_text_manifest.jsonl exists in the data folder. This 
    file is a clean manifest of which card images we have tags for, and which
    training dataset they belong to.
2. Review the src/config.py file for relevant parameters for this training set.
    Should only be USE_GRANITE or not (default False is fine).
"""

# ---------- packages, config, and data ---------------------------------------

## packages

### file manipulation
import json

### link directory
from pathlib import Path
import sys

workspace_root = Path.cwd()
if workspace_root.name == 'notebooks':
    workspace_root = workspace_root.parent
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

### custom packages
from src.ocr.input_dataset import load_manifest_records, summarize_manifest
from src.ocr.dataset_generator import DoclingCardTextDatasetGenerator

## config
from src.config import USE_GRANITE, OCR_FILENAME

## data
manifest_path = workspace_root / "data" / "card_image_text_manifest.jsonl"
records = load_manifest_records(manifest_path, skip_missing_images=True)
summary = summarize_manifest(records)
print(f'Input Data Summary:')
for k, v in summary.items():
    print(f'\t{k}: {v}')

# ---------- generate ocr text ------------------------------------------------

dataset_generator = DoclingCardTextDatasetGenerator(
    data = records,
    use_granite = USE_GRANITE
)
dataset_generator.run()

# ---------- save output ------------------------------------------------------

out_path = workspace_root / "data" / OCR_FILENAME
with open(out_path, 'w', encoding = 'utf-8') as f:
    json.dump(dataset_generator.generated_text, f, indent = 4)
print(f'Docling generated card text saved to {str(out_path)}')

# -----------------------------------------------------------------------------