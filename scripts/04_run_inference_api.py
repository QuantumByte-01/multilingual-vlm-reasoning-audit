"""
Script 04b — Run Inference (API Models — Gemini, GPT-4o, Qwen-72B)
====================================================================
Runs Gemma3, GPT-4o, and Qwen2.5-VL-72B on the translated dataset.
Can run locally (no GPU needed) since these are cloud APIs.

Rate limits:
  - Gemma3-27B       : 30 req/min (Google AI free tier)
  - GPT-4o           : depends on tier; use --batch for Batch API
  - Qwen2.5-VL-72B   : 60 req/min (Together AI, $0.12/$0.39 per 1M in/out)

Output:
  results/raw/{model_display_name}.jsonl

Usage:
  python scripts/04_run_inference_api.py --model gemma3 --lang all
  python scripts/04_run_inference_api.py --model gpt4o --lang all --batch   # OpenAI Batch API
  python scripts/04_run_inference_api.py --model qwen-72b --lang all        # Together AI
  python scripts/04_run_inference_api.py --model all --lang all
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LANGUAGES = ["en", "hi", "ta", "te", "bn", "kn", "mr"]

API_MODELS = {
    "gemini-flash": {"name": "gemini-2.5-flash",                      "rpm": 15, "provider": "google"},
    "gpt4o":        {"name": "gpt-4o",                                 "rpm": 60, "provider": "openai"},
    "gemma3":       {"name": "google/gemma-3-27b-it",                  "rpm": 60, "provider": "deepinfra"},
    "qwen-72b":     {"name": "Qwen/Qwen2.5-VL-72B-Instruct",          "rpm": 60, "provider": "together"},
    "qwen-32b":     {"name": "Qwen/Qwen2.5-VL-32B-Instruct",          "rpm": 60, "provider": "deepinfra"},
    "qwen3-vl-30b":      {"name": "Qwen/Qwen3-VL-30B-A3B-Instruct",                        "rpm": 60, "provider": "deepinfra"},
    "llama4-maverick":   {"name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",   "rpm": 60, "provider": "deepinfra"},
    # Ablation models — qwen-7b not on DeepInfra; run via qwen-32b on deepinfra
    "qwen-7b":           {"name": "Qwen/Qwen2.5-VL-32B-Instruct",                        "rpm": 60, "provider": "deepinfra"},
}

# Human-readable display names for JSONL filenames
MODEL_DISPLAY_NAMES = {
    "gemini-flash": "gemini-2.5-flash",
    "gpt4o":        "gpt-4o",
    "gemma3":       "gemma3-27b",
    "qwen-72b":     "qwen2.5-vl-72b",
    "qwen-32b":     "qwen2.5-vl-32b",
    "qwen3-vl-30b":    "qwen3-vl-30b",
    "llama4-maverick": "llama4-maverick",
    "qwen-7b":         "qwen2.5-vl-7b",
}

# Language metadata for rich output schema
LANG_META = {
    "en": {"family": "germanic",   "script": "latin"},
    "hi": {"family": "indo-aryan", "script": "devanagari"},
    "ta": {"family": "dravidian",  "script": "tamil"},
    "te": {"family": "dravidian",  "script": "telugu"},
    "bn": {"family": "indo-aryan", "script": "bengali"},
    "kn": {"family": "dravidian",  "script": "kannada"},
    "mr": {"family": "indo-aryan", "script": "devanagari"},
}

# ---------------------------------------------------------------------------
# Response analysis helpers (shared with open-source script)
# ---------------------------------------------------------------------------

_ENGLISH_WORD_RE = re.compile(r'\b[a-zA-Z]{3,}\b')

def english_token_ratio(text: str) -> float:
    if not text:
        return 0.0
    words = text.split()
    eng = len(_ENGLISH_WORD_RE.findall(text))
    return round(eng / max(len(words), 1), 3)


def detect_response_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "unknown"


def extract_answer(response: str, answer_type: str, options: list, ground_truth: str = "") -> str | None:
    resp = response.strip()
    # Boolean yes/no questions
    gt_lower = ground_truth.strip().lower()
    if gt_lower in ("yes", "no") and not options:
        resp_lower = resp.lower()
        if re.search(r'\byes\b', resp_lower): return "Yes"
        if re.search(r'\bno\b', resp_lower): return "No"
        # A=Yes B=No convention when model picks MCQ letter
        if re.search(r'\bA\b', resp): return "Yes"
        if re.search(r'\bB\b', resp): return "No"
        return None
    if options or answer_type in ("multi_choice", "mcq"):
        m = re.search(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', resp)
        if m:
            return m.group(1)
        found = re.findall(r'[A-D]', resp)
        return found[-1] if found else None
    else:
        for src, tgt in [
            ('०१२३४५६७८९', '0123456789'),
            ('০১২৩৪৫৬৭৮৯', '0123456789'),
            ('௦௧௨௩௪௫௬௭௮௯', '0123456789'),
            ('౦౧౨౩౪౫౬౭౮౯', '0123456789'),
            ('೦೧೨೩೪೫೬೭೮೯', '0123456789'),
        ]:
            for i, ch in enumerate(src):
                response = response.replace(ch, tgt[i])
        m = re.findall(r'[-+]?\d*\.?\d+', response)
        return m[-1] if m else None


def gt_to_letter(ground_truth: str, options: list) -> str | None:
    """If GT is option text (not a letter), map it to the corresponding letter A/B/C/D."""
    gt = ground_truth.strip()
    if len(gt) == 1 and gt.upper() in "ABCDE":
        return gt.upper()
    for i, opt in enumerate(options or []):
        if str(opt).strip().lower() == gt.lower():
            return chr(ord('A') + i)
    return None


def is_correct(extracted: str | None, ground_truth: str, answer_type: str, options: list = None) -> bool | None:
    if extracted is None or not ground_truth:
        return None
    gt = str(ground_truth).strip()
    pred = str(extracted).strip()

    # Yes/No
    if gt.lower() in ("yes", "no"):
        return pred.lower() == gt.lower()

    # If options present, normalise GT to letter then compare
    if options:
        gt_letter = gt_to_letter(gt, options)
        if gt_letter:
            return pred.upper() == gt_letter
        # GT is already a letter
        return pred.upper() == gt.upper()

    if answer_type in ("multi_choice", "mcq") or (len(gt) == 1 and gt.upper() in "ABCDE"):
        return pred.upper() == gt.upper()

    try:
        pf = float(pred)
        gf = float(gt)
        if gf == 0:
            return abs(pf) < 1e-6
        return abs(pf - gf) / abs(gf) <= 0.05
    except (ValueError, TypeError):
        return pred.lower() == gt.lower()


def build_rich_record(
    record: dict,
    lang: str,
    model_display: str,
    raw_response: str,
    latency_sec: float,
    en_records: dict,
) -> dict:
    """Build the rich JSONL record for one inference call."""
    answer_type = record.get("answer_type", "")
    options = record.get("options", [])
    ground_truth = str(record.get("answer", ""))
    extracted = extract_answer(raw_response, answer_type, options, ground_truth)
    correct = is_correct(extracted, ground_truth, answer_type, options)
    resp_lang = detect_response_language(raw_response)
    en_ratio = english_token_ratio(raw_response)
    lmeta = LANG_META.get(lang, {})
    en_rec = en_records.get(record["id"], {})

    return {
        "id": f"{record['id']}_{lang}",
        "dataset": record.get("source", ""),
        "question_id": record["id"],
        "model": model_display,
        "language": lang,
        "language_family": lmeta.get("family", ""),
        "script": lmeta.get("script", ""),
        "input": {
            "image_path": record.get("image_path", ""),
            "question_original_en": en_rec.get("question", ""),
            "question_translated": record.get("question", ""),
            "options": options,
            "ground_truth": ground_truth,
            "question_type": answer_type,
            "reasoning_category": record.get("category", ""),
        },
        "output": {
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "correct": correct,
            "response_language_detected": resp_lang,
            "responded_in_asked_language": (resp_lang == lang or
                                            (lang == "mr" and resp_lang == "hi")),
            "contains_english_tokens": en_ratio > 0,
            "english_token_ratio": en_ratio,
            "response_length_chars": len(raw_response),
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": round(latency_sec, 3),
            "temperature": 0,
            "max_tokens": 512,
            "prompt_template_version": "v1",
            "variant": "cot" if USE_COT else ("no_image" if NO_IMAGE else "standard"),
        },
    }

# Translated instruction suffixes — nudges model to respond in the target language
LANG_INSTRUCTIONS = {
    "en": (
        "Answer with the option letter only (A, B, C, or D) for multiple-choice questions, "
        "or a number for numerical questions."
    ),
    "hi": (
        "बहुविकल्पीय प्रश्नों के लिए केवल विकल्प अक्षर (A, B, C, या D) में उत्तर दें, "
        "या संख्यात्मक प्रश्नों के लिए एक संख्या दें।"
    ),
    "ta": (
        "பல்தேர்வு வினாக்களுக்கு விருப்பக் கடிதம் மட்டும் (A, B, C, அல்லது D) பதில் சொல்லுங்கள், "
        "அல்லது எண் கேள்விகளுக்கு ஒரு எண் கொடுங்கள்."
    ),
    "te": (
        "బహుళ ఎంపిక ప్రశ్నలకు ఎంపిక అక్షరం మాత్రమే (A, B, C, లేదా D) సమాధానం ఇవ్వండి, "
        "లేదా సంఖ్యా ప్రశ్నలకు ఒక సంఖ్య ఇవ్వండి."
    ),
    "bn": (
        "বহুনির্বাচনী প্রশ্নের জন্য শুধুমাত্র বিকল্প অক্ষর (A, B, C, বা D) দিয়ে উত্তর দিন, "
        "অথবা সংখ্যাসূচক প্রশ্নের জন্য একটি সংখ্যা দিন।"
    ),
    "kn": (
        "ಬಹು-ಆಯ್ಕೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಆಯ್ಕೆ ಅಕ್ಷರ ಮಾತ್ರ (A, B, C, ಅಥವಾ D) ಉತ್ತರಿಸಿ, "
        "ಅಥವಾ ಸಂಖ್ಯಾ ಪ್ರಶ್ನೆಗಳಿಗೆ ಒಂದು ಸಂಖ್ಯೆ ನೀಡಿ."
    ),
    "mr": (
        "बहुपर्यायी प्रश्नांसाठी केवळ पर्याय अक्षर (A, B, C, किंवा D) मध्ये उत्तर द्या, "
        "किंवा संख्यात्मक प्रश्नांसाठी एक संख्या द्या."
    ),
}

LANG_INSTRUCTIONS_COT = {
    "en": (
        "Think step by step, then answer with the option letter only (A, B, C, or D) for "
        "multiple-choice questions, or a number for numerical questions."
    ),
    "hi": (
        "चरण दर चरण सोचें, फिर बहुविकल्पीय प्रश्नों के लिए केवल विकल्प अक्षर (A, B, C, या D) में उत्तर दें, "
        "या संख्यात्मक प्रश्नों के लिए एक संख्या दें।"
    ),
    "ta": (
        "படிப்படியாக சிந்தியுங்கள், பின்னர் பல்தேர்வு வினாக்களுக்கு விருப்பக் கடிதம் மட்டும் (A, B, C, அல்லது D) பதில் சொல்லுங்கள், "
        "அல்லது எண் கேள்விகளுக்கு ஒரு எண் கொடுங்கள்."
    ),
    "te": (
        "దశలవారీగా ఆలోచించండి, తర్వాత బహుళ ఎంపిక ప్రశ్నలకు ఎంపిక అక్షరం మాత్రమే (A, B, C, లేదా D) సమాధానం ఇవ్వండి, "
        "లేదా సంఖ్యా ప్రశ్నలకు ఒక సంఖ్య ఇవ్వండి."
    ),
    "bn": (
        "ধাপে ধাপে চিন্তা করুন, তারপর বহুনির্বাচনী প্রশ্নের জন্য শুধুমাত্র বিকল্প অক্ষর (A, B, C, বা D) দিয়ে উত্তর দিন, "
        "অথবা সংখ্যাসূচক প্রশ্নের জন্য একটি সংখ্যা দিন।"
    ),
    "kn": (
        "ಹಂತ ಹಂತವಾಗಿ ಯೋಚಿಸಿ, ನಂತರ ಬಹು-ಆಯ್ಕೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಆಯ್ಕೆ ಅಕ್ಷರ ಮಾತ್ರ (A, B, C, ಅಥವಾ D) ಉತ್ತರಿಸಿ, "
        "ಅಥವಾ ಸಂಖ್ಯಾ ಪ್ರಶ್ನೆಗಳಿಗೆ ಒಂದು ಸಂಖ್ಯೆ ನೀಡಿ."
    ),
    "mr": (
        "पाऊलोपाऊल विचार करा, नंतर बहुपर्यायी प्रश्नांसाठी केवळ पर्याय अक्षर (A, B, C, किंवा D) मध्ये उत्तर द्या, "
        "किंवा संख्यात्मक प्रश्नांसाठी एक संख्या द्या."
    ),
}

# Global flags — set in main() from argparse
NO_IMAGE: bool = False
USE_COT: bool = False


def format_options(options: list[str]) -> str:
    if not options:
        return ""
    lines = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines) + "\n\n"


def build_prompt(record: dict) -> str:
    lang = record.get("lang", record.get("language", "en"))
    opts = format_options(record.get("options") or [])
    instructions = (LANG_INSTRUCTIONS_COT if USE_COT else LANG_INSTRUCTIONS).get(
        lang, LANG_INSTRUCTIONS["en"]
    )
    return f"{record['question']}\n\n{opts}{instructions}"


def load_image_b64(image_path: str, fmt: str = "PNG", quality: int = 85) -> str | None:
    """Load image from disk and return base64-encoded string. Returns None if NO_IMAGE=True."""
    if NO_IMAGE:
        return None
    full_path = ROOT / "data" / "original" / image_path
    if not full_path.exists():
        return None
    img = Image.open(full_path).convert("RGB")
    # Resize to 512×512 max to keep batch file sizes manageable
    img.thumbnail((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

_GEMINI_KEY_IDX = 0  # rotates only on quota errors

def infer_gemini(record: dict, model_name: str) -> str:
    import google.generativeai as genai
    global _GEMINI_KEY_IDX
    all_keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", os.environ["GOOGLE_API_KEY"]).split(",") if k.strip()]

    prompt = build_prompt(record)
    img_b64 = load_image_b64(record.get("image_path", ""))
    gen_cfg = genai.GenerationConfig(temperature=0.0, max_output_tokens=512)

    last_exc = None
    for attempt in range(len(all_keys)):
        key = all_keys[_GEMINI_KEY_IDX % len(all_keys)]
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            if img_b64:
                part_image = {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                resp = model.generate_content([part_image, prompt], generation_config=gen_cfg)
            else:
                resp = model.generate_content(prompt, generation_config=gen_cfg)
            return resp.text.strip()
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "quota" in exc_str or "429" in exc_str or "rate" in exc_str:
                log.warning("Gemini quota on key[%d] — rotating to next key.", _GEMINI_KEY_IDX % len(all_keys))
                _GEMINI_KEY_IDX += 1
                time.sleep(2)  # brief pause before trying next key
            else:
                raise
    raise last_exc


# ---------------------------------------------------------------------------
# GPT-4o
# ---------------------------------------------------------------------------

def infer_openai(record: dict, model_name: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = build_prompt(record)
    img_b64 = load_image_b64(record["image_path"])

    content: list[dict] = []
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Together AI  (OpenAI-compatible, supports Qwen2.5-VL-72B vision)
# ---------------------------------------------------------------------------

def infer_together(record: dict, model_name: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )

    prompt = build_prompt(record)
    img_b64 = load_image_b64(record["image_path"])

    content: list[dict] = []
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# DeepInfra  (OpenAI-compatible, $0.20/$0.60 per 1M in/out)
# ---------------------------------------------------------------------------

def infer_deepinfra(record: dict, model_name: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )

    prompt = build_prompt(record)
    img_b64 = load_image_b64(record["image_path"])

    content: list[dict] = []
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Inference runner with concurrent workers + rate limiting
# ---------------------------------------------------------------------------

def run_inference(
    model_key: str,
    lang: str,
    records: list[dict],
    infer_fn,
    infer_args: tuple,
    rpm: int,
    output_dir: Path,
    en_records: dict,
    workers: int = 1,
) -> None:
    model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    variant_suffix = "-cot" if USE_COT else ("-no_image" if NO_IMAGE else "")
    jsonl_path = output_dir / f"{model_display}{variant_suffix}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect done (question_id, language) pairs from JSONL
    done_keys: set[tuple[str, str]] = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done_keys.add((r["question_id"], r["language"]))
                except Exception:
                    pass

    to_process = [r for r in records if (r["id"], lang) not in done_keys]
    if not to_process:
        log.info("[%s/%s] All %d already done.", model_key, lang, len(records))
        return
    log.info("[%s/%s] Resuming — %d done, %d to go (workers=%d).", model_key, lang,
             len(records) - len(to_process), len(to_process), workers)

    # Token-bucket rate limiter: allows burst up to `workers` but caps sustained rate at rpm
    min_interval = 60.0 / rpm  # min seconds between dispatches
    _dispatch_lock = threading.Lock()
    _last_dispatch = [0.0]

    def throttled_infer(record: dict):
        # Throttle dispatch rate to respect rpm
        with _dispatch_lock:
            now = time.time()
            wait = max(0.0, _last_dispatch[0] + min_interval - now)
            if wait > 0:
                time.sleep(wait)
            _last_dispatch[0] = time.time()

        t0 = time.time()
        try:
            raw_response = infer_fn(record, *infer_args)
        except Exception as exc:
            exc_str = str(exc)
            log.warning("[%s/%s] id=%s error: %s", model_key, lang, record["id"], exc_str)
            if any(kw in exc_str.lower() for kw in ("quota", "429", "rate", "connection", "timeout", "unavailable", "502", "503", "504")):
                log.warning("Transient error — sleeping 30s, record will retry on resume.")
                time.sleep(30)
                return None  # skip writing — stays retryable on resume
            return (record, f"ERROR: {exc_str}", time.time() - t0)
        return (record, raw_response, time.time() - t0)

    write_lock = threading.Lock()
    n_done = 0

    with open(jsonl_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(throttled_infer, r): r for r in to_process}
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue  # transient error, will retry on resume
                record, raw_response, latency = result
                entry = build_rich_record(record, lang, model_display, raw_response, latency, en_records)
                with write_lock:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f_out.flush()
                    n_done += 1
                    if n_done % 50 == 0:
                        log.info("[%s/%s] %d/%d done.", model_key, lang, n_done, len(to_process))

    log.info("[%s/%s] Done. %d results saved → %s", model_key, lang, n_done, jsonl_path)


# ---------------------------------------------------------------------------
# GPT-4o Batch API (50% cheaper, async, no rate limit worries)
# ---------------------------------------------------------------------------

BATCH_CHUNK_SIZE = 80  # ~80 requests × ~900 tokens = ~72k tokens — stays under 90k org limit
# Submit one chunk, wait for completion, then submit next (sequential chunk processing)

def _build_batch_request(record: dict, lang: str, model_name: str) -> dict:
    """Build a single OpenAI Batch API request entry."""
    prompt = build_prompt(record)
    img_b64 = load_image_b64(record["image_path"], fmt="JPEG", quality=40)

    content: list[dict] = []
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
        })
    content.append({"type": "text", "text": prompt})

    return {
        "custom_id": f"{lang}__{record['id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 512,
        },
    }


def run_gpt4o_batch(
    model_key: str,
    model_name: str,
    target_langs: list[str],
    output_dir: Path,
) -> None:
    """Submit chunks of 80 requests sequentially — wait for each to complete before
    submitting next. Avoids the 90k enqueued-token org limit."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)

    batch_dir = ROOT / "results" / "batch_jobs"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Output JSONL — one file for all languages
    out_jsonl = output_dir / f"{model_display}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect done (question_id, language) pairs
    done_keys: set[tuple[str, str]] = set()
    if out_jsonl.exists():
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done_keys.add((r["question_id"], r["language"]))
                except Exception:
                    pass
        log.info("[%s] Resuming — %d already done.", model_key, len(done_keys))

    # Load English records for original question lookup
    en_records = {r["id"]: r for r in load_records("en")}

    for lang in target_langs:
        records = load_records(lang)
        if not records:
            continue

        records_by_id = {r["id"]: r for r in records}
        todo = [r for r in records if (r["id"], lang) not in done_keys]
        if not todo:
            log.info("[%s/%s] All %d records already done.", model_key, lang, len(records))
            continue

        log.info("[%s/%s] %d records to process in chunks of %d ...",
                 model_key, lang, len(todo), BATCH_CHUNK_SIZE)

        chunks = [todo[i:i + BATCH_CHUNK_SIZE] for i in range(0, len(todo), BATCH_CHUNK_SIZE)]

        for chunk_idx, chunk in enumerate(chunks):
            log.info("[%s/%s] Chunk %d/%d (%d requests) ...",
                     model_key, lang, chunk_idx + 1, len(chunks), len(chunk))

            # Build JSONL batch file
            batch_jsonl = batch_dir / f"{model_key}_{lang}_chunk{chunk_idx}.jsonl"
            with open(batch_jsonl, "w", encoding="utf-8") as f:
                for record in chunk:
                    entry = _build_batch_request(record, lang, model_name)
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            file_size_mb = batch_jsonl.stat().st_size / 1e6
            log.info("[%s/%s] JSONL: %.2f MB", model_key, lang, file_size_mb)

            # Upload & submit
            with open(batch_jsonl, "rb") as f:
                batch_file = client.files.create(file=f, purpose="batch")
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            log.info("[%s/%s] Submitted batch %s", model_key, lang, batch.id)

            # Poll until complete
            while True:
                batch = client.batches.retrieve(batch.id)
                counts = batch.request_counts
                log.info("[%s/%s] Chunk %d: %s — %d/%d done",
                         model_key, lang, chunk_idx + 1, batch.status,
                         counts.completed if counts else 0, counts.total if counts else "?")
                if batch.status in ("completed", "failed", "expired", "cancelled"):
                    break
                time.sleep(30)

            # Download and write rich JSONL results
            if batch.status == "completed" and batch.output_file_id:
                content = client.files.content(batch.output_file_id).text
                n_written = 0
                with open(out_jsonl, "a", encoding="utf-8") as f_out:
                    for line in content.strip().split("\n"):
                        if not line:
                            continue
                        result = json.loads(line)
                        custom_id = result["custom_id"]
                        _, rec_id = custom_id.split("__", 1)
                        if result.get("error"):
                            raw_response = f"ERROR: {result['error']}"
                        else:
                            raw_response = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                        src = records_by_id.get(rec_id, {})
                        entry = build_rich_record(src, lang, model_display, raw_response, 0.0, en_records)
                        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        n_written += 1
                log.info("[%s/%s] Chunk %d saved — %d results written", model_key, lang, chunk_idx + 1, n_written)
            else:
                log.warning("[%s/%s] Chunk %d %s — skipping.", model_key, lang, chunk_idx + 1, batch.status)

            batch_jsonl.unlink(missing_ok=True)

        log.info("[%s/%s] Done — all chunks complete.", model_key, lang)

    log.info("[%s/batch] All languages complete.", model_key)


def _download_batch_results(
    client,
    model_key: str,
    lang: str,
    batch_ids: list[str],
    output_dir: Path,
) -> None:
    """Download batch results and merge into responses.json."""
    out_file = output_dir / model_key / lang / "responses.json"

    existing: dict[str, dict] = {}
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["id"]] = r

    records_by_id = {r["id"]: r for r in load_records(lang)}

    for bid in batch_ids:
        batch = client.batches.retrieve(bid)
        if batch.status != "completed" or not batch.output_file_id:
            log.warning("[%s/%s] Batch %s status=%s — skipping.", model_key, lang, bid, batch.status)
            continue

        content = client.files.content(batch.output_file_id).text
        ts = datetime.now(timezone.utc).isoformat()

        for line in content.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            rec_lang, rec_id = custom_id.split("__", 1)

            if result.get("error"):
                response_text = f"ERROR: {result['error']}"
            else:
                response_text = result["response"]["body"]["choices"][0]["message"]["content"].strip()

            src = records_by_id.get(rec_id, {})
            existing[rec_id] = {
                "id": rec_id,
                "source": src.get("source", ""),
                "category": src.get("category", ""),
                "lang": lang,
                "answer_type": src.get("answer_type", ""),
                "answer": src.get("answer", ""),
                "options": src.get("options", []),
                "question": src.get("question", ""),
                "model_response": response_text,
                "timestamp": ts,
            }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(list(existing.values()), f, ensure_ascii=False, indent=2)
    log.info("[%s/%s] Saved %d results to %s", model_key, lang, len(existing), out_file)


def load_records(lang: str) -> list[dict]:
    if lang == "en":
        path = ROOT / "data" / "original" / "dataset.json"
    else:
        path = ROOT / "data" / "translated" / lang / "questions.json"
    if not path.exists():
        log.error("Data not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run API model inference.")
    parser.add_argument("--model", default="all",
                        help="gemini-flash | gpt4o | gemma3 | qwen-32b | qwen-7b | all")
    parser.add_argument("--lang", default="all",
                        help="Comma-separated lang codes or 'all'")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API for gpt4o (50%% cheaper, async, 24h window)")
    parser.add_argument("--cot", action="store_true",
                        help="Chain-of-thought: append 'Think step by step' to prompt")
    parser.add_argument("--no-image", action="store_true",
                        help="Text-only baseline: strip images from all inputs")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent request workers (default 1). Use 5-10 for DeepInfra.")
    args = parser.parse_args()

    global NO_IMAGE, USE_COT
    NO_IMAGE = args.no_image
    USE_COT = args.cot

    # For ablations, exclude main models from 'all' — only run explicitly named models
    if args.model.strip().lower() == "all":
        # 'all' excludes ablation-only model keys (qwen-7b) to avoid redundant re-runs
        target_models = [k for k in API_MODELS if k != "qwen-7b"]
    else:
        target_models = [m.strip() for m in args.model.split(",")]
    target_langs = LANGUAGES if args.lang == "all" \
        else [l.strip() for l in args.lang.split(",")]

    variant = "cot" if USE_COT else ("no_image" if NO_IMAGE else "standard")
    log.info("=" * 60)
    log.info("Step 04b — API Model Inference")
    log.info("Models: %s | Languages: %s | Variant: %s", target_models, target_langs, variant)
    log.info("=" * 60)

    output_dir = ROOT / "results" / "raw"

    for model_key in target_models:
        cfg = API_MODELS.get(model_key)
        if not cfg:
            log.warning("Unknown model: %s — skipping.", model_key)
            continue

        # Pre-load English records for original question lookup
        en_records = {r["id"]: r for r in load_records("en")}

        if cfg["provider"] == "google":
            if "GOOGLE_API_KEY" not in os.environ:
                log.error("GOOGLE_API_KEY not set in .env — skipping %s.", model_key)
                continue
            infer_fn = infer_gemini
            infer_args = (cfg["name"],)
        elif cfg["provider"] == "together":
            if "TOGETHER_API_KEY" not in os.environ:
                log.error("TOGETHER_API_KEY not set in .env — skipping %s.", model_key)
                continue
            infer_fn = infer_together
            infer_args = (cfg["name"],)
        elif cfg["provider"] == "deepinfra":
            if "DEEPINFRA_API_KEY" not in os.environ:
                log.error("DEEPINFRA_API_KEY not set in .env — skipping %s.", model_key)
                continue
            infer_fn = infer_deepinfra
            infer_args = (cfg["name"],)
        else:
            if "OPENAI_API_KEY" not in os.environ:
                log.error("OPENAI_API_KEY not set in .env — skipping %s.", model_key)
                continue
            # Batch API path for gpt4o
            if args.batch and model_key == "gpt4o":
                log.info("[%s] Using OpenAI Batch API (50%% cheaper, async).", model_key)
                run_gpt4o_batch(model_key, cfg["name"], target_langs, output_dir)
                continue
            infer_fn = infer_openai
            infer_args = (cfg["name"],)

        for lang in target_langs:
            records = load_records(lang)
            if not records:
                continue
            run_inference(
                model_key, lang, records, infer_fn, infer_args,
                rpm=cfg["rpm"], output_dir=output_dir, en_records=en_records,
                workers=args.workers,
            )

    log.info("=" * 60)
    log.info("API inference complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
