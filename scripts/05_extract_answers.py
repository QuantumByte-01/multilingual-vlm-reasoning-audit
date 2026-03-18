"""
Script 05 — Extract Structured Answers from Rich JSONL Results
==============================================================
Reads the flat results/raw/<model>.jsonl files (new format produced by
04_run_inference_*.py).  Each JSONL record already contains
output.extracted_answer and output.correct; this script re-exports them
into results/processed/<model>/<lang>/extracted.json in the format that
script 06 expects, and logs extraction failure rates per model/language.

Output per model × language:
  results/processed/<model>/<lang>/extracted.json
    [{"id", "source", "category", "lang", "answer_type", "answer",
      "predicted", "correct", "extraction_failed", "model_response"}, ...]

Usage:
  python scripts/05_extract_answers.py
  python scripts/05_extract_answers.py --model gpt-4o --lang hi,ta
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LANGUAGES = ["en", "hi", "ta", "te", "bn", "kn", "mr"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def jsonl_record_to_extracted(r: dict) -> dict:
    """Convert a rich JSONL record to the extracted.json schema script 06 expects."""
    inp = r.get("input", {})
    out = r.get("output", {})
    return {
        "id":                r.get("question_id", r.get("id", "")),
        "source":            r.get("dataset", ""),
        "category":          inp.get("reasoning_category", ""),
        "lang":              r.get("language", ""),
        "answer_type":       inp.get("question_type", ""),
        "answer":            inp.get("ground_truth", ""),
        "predicted":         out.get("extracted_answer"),
        "correct":           out.get("correct", False) or False,
        "extraction_failed": out.get("extracted_answer") is None,
        "model_response":    out.get("raw_response", ""),
        # Extra fields used for ablation analysis
        "response_language_detected":   out.get("response_language_detected", ""),
        "responded_in_asked_language":  out.get("responded_in_asked_language", None),
        "english_token_ratio":          out.get("english_token_ratio", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert rich JSONL to per-lang extracted.json.")
    parser.add_argument("--model", default="all",
                        help="Model display name substring, or 'all'")
    parser.add_argument("--lang", default="all",
                        help="Comma-separated lang codes, or 'all'")
    args = parser.parse_args()

    raw_dir = ROOT / "results" / "raw"
    processed_dir = ROOT / "results" / "processed"

    target_langs = LANGUAGES if args.lang.strip().lower() == "all" \
        else [l.strip() for l in args.lang.split(",")]

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No .jsonl files found in %s — run inference first.", raw_dir)
        sys.exit(1)

    if args.model.strip().lower() != "all":
        filter_name = args.model.strip().lower()
        jsonl_files = [f for f in jsonl_files if filter_name in f.stem.lower()]

    log.info("=" * 60)
    log.info("Step 05 — Extract Answers (JSONL → processed/)")
    log.info("Found %d JSONL file(s): %s", len(jsonl_files),
             [f.name for f in jsonl_files])
    log.info("=" * 60)

    for jsonl_path in jsonl_files:
        model_display = jsonl_path.stem  # e.g. "gpt-4o", "qwen2.5-vl-7b"
        log.info("Processing %s ...", jsonl_path.name)

        records = load_jsonl(jsonl_path)
        if not records:
            log.warning("  Empty file — skipping.")
            continue

        # Group by language
        by_lang: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            lang = r.get("language", "")
            if lang:
                by_lang[lang].append(r)

        for lang, lang_records in by_lang.items():
            if lang not in target_langs:
                continue

            extracted = [jsonl_record_to_extracted(r) for r in lang_records]

            n = len(extracted)
            n_failed = sum(1 for r in extracted if r["extraction_failed"])
            n_correct = sum(1 for r in extracted if r["correct"])
            acc = n_correct / (n - n_failed) * 100 if (n - n_failed) > 0 else 0.0

            out_dir = processed_dir / model_display / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "extracted.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            log.info(
                "  [%s/%s] total=%d  failed=%d  correct=%d  acc=%.1f%%",
                model_display, lang, n, n_failed, n_correct, acc,
            )

    log.info("=" * 60)
    log.info("Extraction complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
