"""
Script — Upload Translated Dataset to HuggingFace Hub
======================================================
Uploads the translated benchmark dataset to your HuggingFace account
as a public dataset for reproducibility and citation.

What gets uploaded:
  - data/original/dataset.json            (English filtered questions)
  - data/original/images/                 (all question images)
  - data/translated/<lang>/questions.json (6 Indian language translations)
  - data/verification/agreement_scores.json

The dataset will be available at:
  https://huggingface.co/datasets/<HF_USERNAME>/multilingual-vlm-reasoning

Requirements:
  pip install huggingface-hub
  Set HF_TOKEN in .env (write-access token from hf.co/settings/tokens)

Usage:
  python scripts/upload_to_hf.py --repo your-hf-username/multilingual-vlm-reasoning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LANGUAGES = ["hi", "ta", "te", "bn", "kn", "mr"]


def upload_dataset(repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=token)
        log.info("Created HuggingFace dataset repo: %s", repo_id)
    except Exception:
        log.info("Repo %s already exists — uploading files.", repo_id)

    data_dir = ROOT / "data"

    # Upload English dataset
    en_file = data_dir / "original" / "dataset.json"
    if en_file.exists():
        api.upload_file(
            path_or_fileobj=str(en_file),
            path_in_repo="data/en/dataset.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        log.info("Uploaded English dataset.json")

    # Upload images as a folder
    images_dir = data_dir / "original" / "images"
    if images_dir.exists():
        api.upload_folder(
            folder_path=str(images_dir),
            path_in_repo="data/images",
            repo_id=repo_id,
            repo_type="dataset",
        )
        log.info("Uploaded %d images.", len(list(images_dir.glob("*.png"))))

    # Upload translations
    for lang in LANGUAGES:
        q_file = data_dir / "translated" / lang / "questions.json"
        if q_file.exists():
            api.upload_file(
                path_or_fileobj=str(q_file),
                path_in_repo=f"data/{lang}/questions.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
            log.info("Uploaded %s translations.", lang)
        else:
            log.warning("No translation found for %s — skipping.", lang)

    # Upload verification scores
    scores_file = data_dir / "verification" / "agreement_scores.json"
    if scores_file.exists():
        api.upload_file(
            path_or_fileobj=str(scores_file),
            path_in_repo="verification/agreement_scores.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        log.info("Uploaded agreement scores.")

    # Upload a dataset card (README.md for HF)
    card_content = f"""---
license: cc-by-4.0
language:
- en
- hi
- ta
- te
- bn
- kn
- mr
tags:
- visual-reasoning
- multilingual
- indian-languages
- vlm-evaluation
- mathvista
- scienceqa
- mmmu
pretty_name: Multilingual VLM Visual Reasoning Benchmark (Indian Languages)
size_categories:
- 1K<n<10K
---

# Multilingual VLM Visual Reasoning Benchmark

**Paper:** Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages

## Overview

This dataset contains ~1,000 visual reasoning questions translated from English into 6 Indian languages:
Hindi (hi), Tamil (ta), Telugu (te), Bengali (bn), Kannada (kn), Marathi (mr).

**Source benchmarks:** MathVista (testmini), ScienceQA, MMMU
**Translation:** IndicTrans2 (AI4Bharat), verified against GPT-4o/Gemini

## Dataset Structure

```
data/
  en/dataset.json          # English filtered questions
  hi/questions.json        # Hindi translations
  ta/questions.json        # Tamil translations
  te/questions.json        # Telugu translations
  bn/questions.json        # Bengali translations
  kn/questions.json        # Kannada translations
  mr/questions.json        # Marathi translations
  images/                  # Question images (PNG)
verification/
  agreement_scores.json    # Inter-translator agreement (IndicTrans2 vs GPT-4o)
```

## Record Schema

```json
{{
  "id": "mathvista_0001",
  "source": "mathvista",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "answer": "B",
  "answer_type": "mcq",
  "category": "math",
  "image_path": "images/mathvista_0001.png",
  "lang": "hi",
  "original_question": "..."
}}
```

## Citation

If you use this dataset, please cite our paper.
"""

    card_path = ROOT / "data" / "HF_DATASET_CARD.md"
    card_path.write_text(card_content, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info("Uploaded dataset card.")
    log.info("Dataset available at: https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub.")
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID, e.g. your-username/multilingual-vlm-reasoning",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set in .env. Get a write token from hf.co/settings/tokens")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Uploading dataset to HuggingFace: %s", args.repo)
    log.info("=" * 60)

    upload_dataset(args.repo, token)

    log.info("=" * 60)
    log.info("Upload complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
