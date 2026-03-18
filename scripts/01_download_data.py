"""
Script 01 — Download and Filter Datasets
=========================================
Downloads MathVista (testmini), ScienceQA (image subset), and MMMU (STEM subset)
from HuggingFace, applies filtering criteria, and saves:
  - data/original/dataset.json  : metadata (no images embedded)
  - data/original/images/       : one PNG per question ID

Filtering rules:
  - MathVista  : Remove pure visual pattern-matching tasks (no translatable text).
  - ScienceQA  : Keep has_image=True AND subject="natural science" only.
  - MMMU       : Keep STEM subjects (Math, Physics, Chemistry, Engineering, CS).

Output record schema (dataset.json):
  {
    "id"          : str,   # unique ID, e.g. "mathvista_0001"
    "source"      : str,   # "mathvista" | "scienceqa" | "mmmu"
    "question"    : str,   # question text in English
    "options"     : list,  # answer options (empty list for free-form)
    "answer"      : str,   # ground-truth answer (letter or value)
    "answer_type" : str,   # "mcq" | "free_form"
    "category"    : str,   # "math" | "science" | "stem"
    "image_path"  : str    # relative path to the image PNG
  }

Usage:
  uv run python scripts/01_download_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "original"
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_FILE = OUTPUT_DIR / "dataset.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MathVista task skills that contain no translatable question text
MATHVISTA_REMOVE_SKILLS = {
    "pattern reasoning",
    "spatial reasoning",
    "visual reasoning",
}

# MMMU STEM subjects to download
MMMU_KEEP_SUBJECTS = {
    "Math",
    "Physics",
    "Chemistry",
    "Architecture_and_Engineering",  # "Engineering" was renamed in MMMU v2
    "Mechanical_Engineering",
    "Computer_Science",
}

TARGET_MATHVISTA = 400
TARGET_SCIENCEQA = 400
TARGET_MMMU = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_pil_image(image: Image.Image, path: Path) -> bool:
    """Save a PIL image to disk as PNG. Returns True on success."""
    try:
        image.convert("RGB").save(path, format="PNG")
        return True
    except Exception as exc:
        log.warning("Could not save image to %s: %s", path, exc)
        return False


def has_translatable_text(question: str, min_words: int = 4) -> bool:
    """Return True if the question has enough alphabetic words to translate."""
    words = [w for w in question.strip().split() if w.isalpha()]
    return len(words) >= min_words


# ---------------------------------------------------------------------------
# MathVista
# ---------------------------------------------------------------------------

def load_mathvista() -> list[dict[str, Any]]:
    """Load and filter MathVista testmini split."""
    log.info("Loading MathVista (testmini) ...")
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    log.info("  Raw size: %d", len(ds))

    records: list[dict[str, Any]] = []
    skipped_skill = 0
    skipped_text = 0
    skipped_image = 0

    for row in tqdm(ds, desc="MathVista", unit="q"):
        # Filter 1: remove pure-visual task types
        skills = [s.lower() for s in (row.get("skills") or [])]
        if any(s in MATHVISTA_REMOVE_SKILLS for s in skills):
            skipped_skill += 1
            continue

        question = str(row.get("question") or "").strip()

        # Filter 2: question must have translatable text
        if not has_translatable_text(question):
            skipped_text += 1
            continue

        pid = str(row.get("pid", len(records)))
        record_id = f"mathvista_{pid}"

        # Save image to disk — MathVista stores PIL image in 'decoded_image'
        img = row.get("decoded_image") or row.get("image")
        image_path = ""
        if img is not None:
            img_file = IMAGE_DIR / f"{record_id}.png"
            if save_pil_image(img, img_file):
                image_path = f"images/{record_id}.png"
            else:
                skipped_image += 1
                continue
        else:
            skipped_image += 1
            continue  # skip questions without images

        # Build options list
        options = row.get("choices") or []
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                options = [options]

        answer = str(row.get("answer") or "").strip()
        answer_type = "mcq" if options else "free_form"

        records.append({
            "id": record_id,
            "source": "mathvista",
            "question": question,
            "options": options,
            "answer": answer,
            "answer_type": answer_type,
            "category": "math",
            "image_path": image_path,
        })

    log.info(
        "  MathVista done: kept=%d  skipped_skill=%d  skipped_text=%d  skipped_image=%d",
        len(records), skipped_skill, skipped_text, skipped_image,
    )
    return records


# ---------------------------------------------------------------------------
# ScienceQA
# ---------------------------------------------------------------------------

def load_scienceqa() -> list[dict[str, Any]]:
    """Load and filter ScienceQA (image + natural science subset)."""
    log.info("Loading ScienceQA (ScienceQA-IMG, test split) ...")
    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test")
    log.info("  Raw size: %d", len(ds))

    records: list[dict[str, Any]] = []
    skipped_no_image = 0
    skipped_subject = 0

    for idx, row in enumerate(tqdm(ds, desc="ScienceQA", unit="q")):
        # Filter 1: must have image
        img = row.get("image")
        if img is None:
            skipped_no_image += 1
            continue

        # Filter 2: natural science only
        subject = str(row.get("subject") or "").lower()
        if "natural" not in subject:
            skipped_subject += 1
            continue

        question = str(row.get("question") or "").strip()
        options = list(row.get("choices") or [])
        answer_idx = row.get("answer")

        if isinstance(answer_idx, int) and options:
            answer = chr(ord("A") + answer_idx)
        else:
            answer = str(answer_idx or "")

        record_id = f"scienceqa_{idx:05d}"
        img_file = IMAGE_DIR / f"{record_id}.png"

        if not save_pil_image(img, img_file):
            continue
        image_path = f"images/{record_id}.png"

        records.append({
            "id": record_id,
            "source": "scienceqa",
            "question": question,
            "options": options,
            "answer": answer,
            "answer_type": "mcq",
            "category": "science",
            "image_path": image_path,
        })

    log.info(
        "  ScienceQA done: kept=%d  skipped_no_image=%d  skipped_subject=%d",
        len(records), skipped_no_image, skipped_subject,
    )
    return records


# ---------------------------------------------------------------------------
# MMMU
# ---------------------------------------------------------------------------

def load_mmmu() -> list[dict[str, Any]]:
    """Load and filter MMMU validation split (STEM subjects only)."""
    log.info("Loading MMMU (STEM subjects, validation split) ...")

    records: list[dict[str, Any]] = []
    total_raw = 0

    for subject in tqdm(MMMU_KEEP_SUBJECTS, desc="MMMU subjects"):
        try:
            ds = load_dataset("MMMU/MMMU", name=subject, split="validation")
        except Exception as exc:
            log.warning("  Could not load MMMU subject '%s': %s", subject, exc)
            continue

        total_raw += len(ds)

        for idx, row in enumerate(ds):
            question = str(row.get("question") or "").strip()
            options_raw = row.get("options") or []

            if isinstance(options_raw, str):
                try:
                    options_raw = json.loads(options_raw)
                except json.JSONDecodeError:
                    options_raw = [options_raw]

            options = [str(o) for o in options_raw]
            answer = str(row.get("answer") or "").strip()
            answer_type = "mcq" if options else "free_form"

            record_id = f"mmmu_{subject.lower()}_{idx:04d}"

            # MMMU may have image_1 through image_7
            image_path = ""
            for key in ("image_1", "image_2", "image"):
                img = row.get(key)
                if img is not None:
                    img_file = IMAGE_DIR / f"{record_id}.png"
                    if save_pil_image(img, img_file):
                        image_path = f"images/{record_id}.png"
                    break

            if not image_path:
                continue  # skip questions without images

            records.append({
                "id": record_id,
                "source": "mmmu",
                "question": question,
                "options": options,
                "answer": answer,
                "answer_type": answer_type,
                "category": "stem",
                "image_path": image_path,
            })

    log.info(
        "  MMMU done: total_raw=%d  kept=%d",
        total_raw, len(records),
    )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Step 01 — Download and Filter Datasets")
    log.info("=" * 60)

    mathvista_records = load_mathvista()
    scienceqa_records = load_scienceqa()
    mmmu_records = load_mmmu()

    # Cap to target counts
    mathvista_records = mathvista_records[:TARGET_MATHVISTA]
    scienceqa_records = scienceqa_records[:TARGET_SCIENCEQA]
    mmmu_records = mmmu_records[:TARGET_MMMU]

    all_records = mathvista_records + scienceqa_records + mmmu_records

    log.info("")
    log.info("Final dataset summary:")
    log.info("  MathVista : %d questions", len(mathvista_records))
    log.info("  ScienceQA : %d questions", len(scienceqa_records))
    log.info("  MMMU      : %d questions", len(mmmu_records))
    log.info("  TOTAL     : %d questions", len(all_records))
    log.info("")

    # Validate no duplicate IDs
    ids = [r["id"] for r in all_records]
    assert len(ids) == len(set(ids)), "Duplicate IDs found!"

    # Print 2 random samples from each dataset for verification
    import random
    random.seed(42)
    log.info("--- Sample records (2 per dataset) ---")
    for source in ("mathvista", "scienceqa", "mmmu"):
        subset = [r for r in all_records if r["source"] == source]
        samples = random.sample(subset, min(2, len(subset)))
        for s in samples:
            log.info(
                "[%s] id=%s | q=%s | ans=%s | type=%s",
                source, s["id"],
                s["question"][:80].replace("\n", " "),
                s["answer"], s["answer_type"],
            )
    log.info("--------------------------------------")

    log.info("Saving dataset.json to %s ...", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    log.info(
        "Done. dataset.json size: %.1f MB | images saved: %d",
        OUTPUT_FILE.stat().st_size / 1_048_576,
        len(list(IMAGE_DIR.glob("*.png"))),
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
