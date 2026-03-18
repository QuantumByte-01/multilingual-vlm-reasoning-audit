"""
Script 02 — Translate Dataset into 6 Indian Languages
======================================================
Uses IndicTrans2 (AI4Bharat) to translate all English questions into:
  hi (Hindi), ta (Tamil), te (Telugu), bn (Bengali), kn (Kannada), mr (Marathi)

Designed to run on a GPU instance (AMD MI300X / NVIDIA A100).
Falls back to CPU automatically but will be ~60x slower.

Translation decisions (documented in paper):
  - Math symbols (π, √, =, +)  → kept as-is
  - Numbers (42, 3.14)          → kept (Arabic numerals)
  - Units (km, m/s, °C)         → kept in English
  - Answer options (A, B, C, D) → kept as-is
  - Technical terms             → translated via IndicTrans2
  - Instruction text            → fully translated

Output:
  data/translated/<lang>/questions.json   one JSON per language

Usage (GPU instance):
  python scripts/02_translate.py --lang all
  python scripts/02_translate.py --lang hi,ta
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language config
# ---------------------------------------------------------------------------
LANGUAGES = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
}
SRC_LANG = "eng_Latn"

BATCH_SIZE = 32          # reduce to 8 on small GPU
MAX_RETRIES = 3
RETRY_DELAY = 5          # seconds


# ---------------------------------------------------------------------------
# Pre/post-processing — preserve math tokens
# ---------------------------------------------------------------------------

# Tokens that must not be translated
# FIX: added leading \b to units group — prevents matching single letters
# (g, s, m, ...) at the END of English words like "engineering", "process".
_PRESERVE_PATTERN = re.compile(
    r"(\b[A-D]\b"                   # MCQ option labels
    r"|\b\d+(?:[.,]\d+)*\b"         # numbers
    r"|[+\-×÷=≠≤≥π√∑∫∞°]"         # math symbols (removed <> to avoid false matches)
    r"|\b(?:km|m|cm|mm|kg|g|mg|s|ms|°C|°F|K|m\/s|km\/h|Hz|kHz|MHz|GHz|W|kW|J|N|Pa|atm|mol|L|mL|μm|nm|μg|μL)\b"  # units (FIXED: leading \b)
    r")"
)

# Placeholder format: @@N@@ — ASCII digits only, no letters to transliterate.
# IndicTrans2 copies unknown ASCII punctuation+digit sequences through unchanged.
_PH_FMT = "@@{}@@"
_PH_RESTORE_VARIANTS = [
    "@@{}@@",      # exact match
    "@@ {} @@",    # IndicProcessor may add spaces inside
    "@ @{}@ @",    # further spaced variant
]


def _tokenize_preserving(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace preserved tokens with @@N@@ placeholders.
    Returns the modified text and the {placeholder: original_token} mapping.
    """
    placeholders: dict[str, str] = {}
    counter = [0]

    def replace(match: re.Match) -> str:
        token = match.group(0)
        ph = _PH_FMT.format(counter[0])
        placeholders[ph] = token
        counter[0] += 1
        return ph

    modified = _PRESERVE_PATTERN.sub(replace, text)
    return modified, placeholders


def _restore_placeholders(translated: str, placeholders: dict[str, str]) -> str:
    """Restore @@N@@ placeholders, handling spacing variants IndicProcessor may add."""
    for ph, original in placeholders.items():
        n = ph[2:-2]  # digit(s) between @@...@@
        if ph in translated:
            translated = translated.replace(ph, original)
        else:
            for variant_fmt in _PH_RESTORE_VARIANTS[1:]:
                variant = variant_fmt.format(n)
                if variant in translated:
                    translated = translated.replace(variant, original)
                    break
    return translated


def preprocess(text: str) -> tuple[str, dict[str, str]]:
    return _tokenize_preserving(text)


def postprocess(translated: str, placeholders: dict[str, str]) -> str:
    return _restore_placeholders(translated, placeholders)


# ---------------------------------------------------------------------------
# IndicTrans2 model loader
# ---------------------------------------------------------------------------

def load_indictrans2():
    """Load IndicTrans2 en→indic model, tokenizer, and IndicProcessor."""
    log.info("Loading IndicTrans2 (ai4bharat/indictrans2-en-indic-1B) ...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        log.error("Missing dependencies. Run: pip install transformers sentencepiece IndicTransToolkit")
        sys.exit(1)

    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")

    model_name = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    processor = IndicProcessor(inference=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU detected — translation will be slow on CPU.")
    else:
        log.info("GPU detected: %s", torch.cuda.get_device_name(0))

    model = model.to(device)
    model.eval()

    log.info("IndicTrans2 loaded on %s.", device.upper())
    return tokenizer, model, processor, device


def translate_batch(
    texts: list[str],
    tgt_lang: str,
    tokenizer,
    model,
    processor,
    device: str,
) -> list[str]:
    """Translate a batch of texts from English to tgt_lang using IndicTransToolkit."""
    # Pre-process: protect math tokens before IndicProcessor
    pairs = [preprocess(t) for t in texts]
    processed_texts = [p[0] for p in pairs]
    maps = [p[1] for p in pairs]

    # IndicProcessor preprocessing (handles script normalization, tokenization)
    batch = processor.preprocess_batch(processed_texts, src_lang=SRC_LANG, tgt_lang=tgt_lang)

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_length=256,
            early_stopping=True,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # IndicProcessor postprocessing
    translations = processor.postprocess_batch(decoded, lang=tgt_lang)

    # Restore math token placeholders
    restored = [postprocess(t, m) for t, m in zip(translations, maps)]
    return restored


# ---------------------------------------------------------------------------
# Translation fields
# ---------------------------------------------------------------------------

def fields_to_translate(record: dict) -> dict[str, str]:
    """Return {field_name: text} for all fields that need translation."""
    out = {"question": record["question"]}
    for i, opt in enumerate(record.get("options") or []):
        out[f"option_{i}"] = opt
    return out


def apply_translations(record: dict, translated: dict[str, str]) -> dict:
    """Build a translated record by merging translations into the original."""
    new = dict(record)
    new["question"] = translated["question"]
    new_options = []
    for i in range(len(record.get("options") or [])):
        key = f"option_{i}"
        new_options.append(translated.get(key, record["options"][i]))
    new["options"] = new_options
    return new


# ---------------------------------------------------------------------------
# Per-language translation
# ---------------------------------------------------------------------------

def translate_language(
    records: list[dict],
    lang_code: str,
    tokenizer,
    model,
    processor,
    device: str,
    output_dir: Path,
) -> None:
    tgt_lang = LANGUAGES[lang_code]
    out_file = output_dir / lang_code / "questions.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-translated records
    existing: dict[str, dict] = {}
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["id"]] = r
        log.info("[%s] Resuming — %d already translated.", lang_code, len(existing))

    to_translate = [r for r in records if r["id"] not in existing]
    log.info("[%s] Need to translate %d records.", lang_code, len(to_translate))

    results: list[dict] = list(existing.values())

    # Collect all (record_idx, field_name, text) triples
    flat_texts: list[tuple[int, str, str]] = []
    for i, record in enumerate(to_translate):
        for field, text in fields_to_translate(record).items():
            flat_texts.append((i, field, text))

    # Batch-translate all texts at once
    all_translated: list[str] = []
    for start in tqdm(
        range(0, len(flat_texts), BATCH_SIZE),
        desc=f"Translating [{lang_code}]",
        unit="batch",
    ):
        batch_triples = flat_texts[start : start + BATCH_SIZE]
        batch_texts = [t for _, _, t in batch_triples]

        for attempt in range(MAX_RETRIES):
            try:
                translated = translate_batch(batch_texts, tgt_lang, tokenizer, model, processor, device)
                all_translated.extend(translated)
                break
            except Exception as exc:
                log.warning(
                    "[%s] Batch %d failed (attempt %d/%d): %s",
                    lang_code, start // BATCH_SIZE, attempt + 1, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    # Fallback: use original English on failure
                    log.error("[%s] Batch failed permanently. Using English fallback.", lang_code)
                    all_translated.extend(batch_texts)

    # Reconstruct records from flat translations
    # Map: record_idx -> {field: translated_text}
    record_translations: dict[int, dict[str, str]] = {}
    for (rec_idx, field, _original), translation in zip(flat_texts, all_translated):
        if rec_idx not in record_translations:
            record_translations[rec_idx] = {}
        record_translations[rec_idx][field] = translation

    for i, record in enumerate(to_translate):
        translated_record = apply_translations(record, record_translations.get(i, {}))
        translated_record["lang"] = lang_code
        translated_record["original_question"] = record["question"]
        results.append(translated_record)

    # Save
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info("[%s] Saved %d records to %s", lang_code, len(results), out_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description="Translate dataset into Indian languages.")
    parser.add_argument(
        "--lang",
        default="all",
        help="Comma-separated language codes (e.g. hi,ta) or 'all'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for translation (default: 32)",
    )
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    # Determine which languages to translate
    if args.lang.strip().lower() == "all":
        target_langs = list(LANGUAGES.keys())
    else:
        target_langs = [l.strip() for l in args.lang.split(",") if l.strip() in LANGUAGES]
        invalid = [l.strip() for l in args.lang.split(",") if l.strip() not in LANGUAGES]
        if invalid:
            log.warning("Unknown language codes: %s — skipping.", invalid)

    log.info("=" * 60)
    log.info("Step 02 — Translate Dataset")
    log.info("Target languages: %s", target_langs)
    log.info("=" * 60)

    # Load English dataset
    dataset_file = ROOT / "data" / "original" / "dataset.json"
    if not dataset_file.exists():
        log.error("dataset.json not found. Run 01_download_data.py first.")
        sys.exit(1)

    with open(dataset_file, encoding="utf-8") as f:
        records = json.load(f)
    log.info("Loaded %d English records.", len(records))

    # Load model once — reuse for all languages
    tokenizer, model, processor, device = load_indictrans2()

    translated_dir = ROOT / "data" / "translated"

    for lang_code in target_langs:
        log.info("-" * 50)
        log.info("Translating to [%s] (%s) ...", lang_code, LANGUAGES[lang_code])
        t0 = time.time()
        translate_language(records, lang_code, tokenizer, model, processor, device, translated_dir)
        elapsed = time.time() - t0
        log.info("[%s] Finished in %.1f minutes.", lang_code, elapsed / 60)

    log.info("=" * 60)
    log.info("All translations complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
