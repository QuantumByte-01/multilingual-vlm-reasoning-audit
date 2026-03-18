"""
Script 03 — Verify Translation Quality
=======================================
Cross-checks IndicTrans2 translations against a secondary translator
(GPT-4o or Gemini) on a random sample of 100 questions per language.

Outputs:
  data/verification/agreement_scores.json   — agreement stats per language
  data/verification/<lang>_sample.json      — per-sample comparison

Usage:
  python scripts/03_verify_translations.py --api gemini --samples 100
  python scripts/03_verify_translations.py --api openai --samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
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
LANGUAGE_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "kn": "Kannada", "mr": "Marathi",
}


# ---------------------------------------------------------------------------
# Secondary translators
# ---------------------------------------------------------------------------

def translate_gemini(texts: list[str], target_lang: str) -> list[str]:
    """Translate via Gemini 2.5 Flash with key rotation across all 7 keys."""
    import google.generativeai as genai
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.gemini_keys import GeminiKeyManager
    km = GeminiKeyManager()

    def _call(key: str, prompt: str) -> str:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()

    results = []
    for text in texts:
        prompt = (
            f"Translate the following English text to {LANGUAGE_NAMES[target_lang]}. "
            f"Keep numbers, math symbols, and option labels (A, B, C, D) unchanged. "
            f"Return ONLY the translation, nothing else.\n\n{text}"
        )
        result = km.call_with_rotation(_call, prompt)
        results.append(result)
        time.sleep(4)    # ~15 RPM free tier; 7 keys = safe throughput with rotation on 429
    return results


def translate_openai(texts: list[str], target_lang: str) -> list[str]:
    """Translate via GPT-4o."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    for text in texts:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a translation engine. Translate English to {LANGUAGE_NAMES[target_lang]}. "
                        "Keep numbers, math symbols (π, √, =, +, etc.), units, and option labels "
                        "(A, B, C, D) unchanged. Return ONLY the translation."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=256,
        )
        results.append(resp.choices[0].message.content.strip())
    return results


# ---------------------------------------------------------------------------
# Agreement metric — character-level Jaccard similarity
# ---------------------------------------------------------------------------

def char_jaccard(a: str, b: str) -> float:
    """Character-level Jaccard similarity between two strings."""
    a_chars = set(a)
    b_chars = set(b)
    if not a_chars and not b_chars:
        return 1.0
    intersection = len(a_chars & b_chars)
    union = len(a_chars | b_chars)
    return intersection / union


def compute_agreement(indictrans_texts: list[str], secondary_texts: list[str]) -> float:
    """Average character Jaccard similarity across all pairs."""
    scores = [char_jaccard(a, b) for a, b in zip(indictrans_texts, secondary_texts)]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify translation quality.")
    parser.add_argument("--api", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--samples", type=int, default=100, help="Samples per language")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    log.info("=" * 60)
    log.info("Step 03 — Verify Translations (secondary: %s)", args.api)
    log.info("=" * 60)

    verify_dir = ROOT / "data" / "verification"
    verify_dir.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, float] = {}

    for lang in LANGUAGES:
        translated_file = ROOT / "data" / "translated" / lang / "questions.json"
        if not translated_file.exists():
            log.warning("[%s] No translation file found — skipping.", lang)
            continue

        with open(translated_file, encoding="utf-8") as f:
            records = json.load(f)

        sample = random.sample(records, min(args.samples, len(records)))
        indictrans_questions = [r["question"] for r in sample]
        original_questions = [r["original_question"] for r in sample]

        log.info("[%s] Getting %s translations for %d samples ...", lang, args.api, len(sample))
        try:
            if args.api == "gemini":
                secondary = translate_gemini(original_questions, lang)
            else:
                secondary = translate_openai(original_questions, lang)
        except Exception as exc:
            log.error("[%s] Secondary translation failed: %s", lang, exc)
            continue

        agreement = compute_agreement(indictrans_questions, secondary)
        all_scores[lang] = round(agreement, 4)
        log.info("[%s] Agreement score: %.4f", lang, agreement)

        # Save per-language comparison
        comparison = [
            {
                "id": r["id"],
                "english": orig,
                "indictrans2": it,
                "secondary": sec,
                "agreement": round(char_jaccard(it, sec), 4),
            }
            for r, orig, it, sec in zip(sample, original_questions, indictrans_questions, secondary)
        ]
        out_file = verify_dir / f"{lang}_sample.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        log.info("[%s] Comparison saved to %s", lang, out_file)

    # Save aggregate scores
    agg_file = verify_dir / "agreement_scores.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    log.info("Agreement scores: %s", all_scores)
    log.info("Saved to %s", agg_file)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
