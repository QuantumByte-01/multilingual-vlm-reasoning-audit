"""
Script 04a — Run Inference (Open-Source Models via vLLM)
=========================================================
Uses vLLM offline batch inference for maximum GPU throughput.
Processes all records in one batch per language — no per-sample Python overhead.

Models supported (6-model final set):
  qwen-7b      : Qwen/Qwen2.5-VL-7B-Instruct       (small open baseline)
  qwen-72b     : Qwen/Qwen2.5-VL-72B-Instruct       (scale: 7B→72B comparison)
  internvl-8b  : OpenGVLab/InternVL2_5-8B           (second family validation)

Dropped (redundant): internvl-26b, internvl-78b, llava, phi-vision, llama4-scout, gemma3
API-only models: gemma3 (Google API), gemini-flash (Google API), gpt4o (OpenAI Batch)

Output:
  results/raw/<model_key>/<lang>/responses.json

Usage:
  python scripts/04_run_inference_open.py --model all --lang all
  python scripts/04_run_inference_open.py --model qwen-7b --lang hi,ta
  python scripts/04_run_inference_open.py --model qwen-72b --lang en
  python scripts/04_run_inference_open.py --model qwen-7b --lang all --no-image   # text-only baseline
  python scripts/04_run_inference_open.py --model qwen-7b --lang all --cot        # chain-of-thought variant
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load env (sets HF_TOKEN, etc.)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Forward HF_TOKEN to both env var names vLLM / HF Hub uses
if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LANGUAGES = ["en", "hi", "ta", "te", "bn", "kn", "mr"]

OPEN_MODELS = {
    # Final GPU models
    "qwen-7b":        "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen-72b":       "Qwen/Qwen2.5-VL-72B-Instruct",
    "internvl-8b":    "OpenGVLab/InternVL2_5-8B",
    "gemma3-27b":     "google/gemma-3-27b-it",
    "qwen3-vl-30b":   "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "aya-vision-8b":  "CohereForAI/aya-vision-8b",
}

# Human-readable names for JSONL filenames
MODEL_DISPLAY_NAMES = {
    "qwen-7b":       "qwen2.5-vl-7b",
    "qwen-72b":      "qwen2.5-vl-72b",
    "internvl-8b":   "internvl2.5-8b",
    "gemma3-27b":    "gemma3-27b",
    "qwen3-vl-30b":  "qwen3-vl-30b",
    "aya-vision-8b": "aya-vision-8b",
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

# Max concurrent sequences — for VLMs with large image tokens, 256 is sufficient
MAX_NUM_SEQS = 256
# For large models, reduce to avoid KV cache fragmentation
MAX_NUM_SEQS_LARGE = 64

# Max batched tokens — overlap prefill and decode for higher throughput
MAX_NUM_BATCHED_TOKENS = 32768

# Max GPU memory utilization — leave 5% headroom on 192GB MI300X
GPU_MEMORY_UTILIZATION = 0.95

# Sampling: deterministic (temperature=0 → greedy)
SAMPLING_PARAMS_CFG = dict(temperature=0.0, max_tokens=512)
# CoT needs more tokens for step-by-step reasoning
SAMPLING_PARAMS_CFG_COT = dict(temperature=0.0, max_tokens=1536)

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

# Global flags set in main()
NO_IMAGE = False
USE_COT = False


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

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


MAX_IMAGE_PIXELS = 512 * 512  # reduced from 768×768 — cuts vision tokens ~55%, faster prefill

# ---------------------------------------------------------------------------
# Response analysis helpers
# ---------------------------------------------------------------------------

_ENGLISH_WORD_RE = re.compile(r'\b[a-zA-Z]{3,}\b')

def english_token_ratio(text: str) -> float:
    """Fraction of whitespace-split tokens that are English words (≥3 chars)."""
    if not text:
        return 0.0
    words = text.split()
    eng = len(_ENGLISH_WORD_RE.findall(text))
    return round(eng / max(len(words), 1), 3)


def detect_response_language(text: str) -> str:
    """Detect language of model response using langdetect."""
    try:
        from langdetect import detect
        return detect(text[:500])  # first 500 chars sufficient
    except Exception:
        return "unknown"


def extract_answer(response: str, answer_type: str, options: list) -> str | None:
    """Basic answer extraction: letter for MCQ, number for free-form."""
    if options or answer_type in ("multi_choice", "mcq"):
        # Look for standalone A/B/C/D
        m = re.search(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', response)
        if m:
            return m.group(1)
        # Last A/B/C/D in response
        found = re.findall(r'[A-D]', response)
        return found[-1] if found else None
    else:
        # Numerical — convert Indian numerals to Arabic, then find last number
        for src, tgt in [
            ('०१२३४५६७८९', '0123456789'),  # Devanagari
            ('০১২৩৪৫৬৭৮৯', '0123456789'),  # Bengali
            ('੦੧੨੩੪੫੬੭੮੯', '0123456789'),  # Punjabi
            ('௦௧௨௩௪௫௬௭௮௯', '0123456789'),  # Tamil
            ('౦౧౨౩౪౫౬౭౮౯', '0123456789'),  # Telugu
            ('೦೧೨೩೪೫೬೭೮೯', '0123456789'),  # Kannada
        ]:
            for i, ch in enumerate(src):
                response = response.replace(ch, tgt[i])
        m = re.findall(r'[-+]?\d*\.?\d+', response)
        return m[-1] if m else None


def is_correct(extracted: str | None, ground_truth: str, answer_type: str) -> bool | None:
    """Check correctness: exact match for MCQ, ±5% for numerical."""
    if extracted is None or not ground_truth:
        return None
    if answer_type in ("multi_choice", "mcq") or len(ground_truth) == 1:
        return extracted.strip().upper() == str(ground_truth).strip().upper()
    try:
        pred = float(extracted)
        gt = float(ground_truth)
        if gt == 0:
            return abs(pred) < 1e-6
        return abs(pred - gt) / abs(gt) <= 0.05
    except (ValueError, TypeError):
        return extracted.strip() == ground_truth.strip()


def load_image(image_path: str) -> Image.Image | None:
    if NO_IMAGE:
        return None  # text-only baseline
    full_path = ROOT / "data" / "original" / image_path
    if not full_path.exists():
        return None
    try:
        img = Image.open(full_path).convert("RGB")
        # Resize if too large — prevents token count explosion in Qwen2-VL / InternVL
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# vLLM model loading
# ---------------------------------------------------------------------------

def load_vllm_model(model_key: str):
    """Load a model with vLLM for maximum GPU throughput."""
    from vllm import LLM

    model_id = OPEN_MODELS[model_key]
    log.info("Loading [%s] via vLLM: %s", model_key, model_id)

    common_kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        # Prefix caching: images are shared across 7 language variants of same question
        # — cache hit on image tokens gives ~2-3x throughput improvement
        enable_prefix_caching=True,
    )

    # Qwen-72B: FP8 to fit in 192GB + more KV cache room; reduce concurrency
    if model_key == "qwen-72b":
        llm = LLM(**{**common_kwargs, "dtype": "fp8", "max_num_seqs": MAX_NUM_SEQS_LARGE})

    # All other models (≤8B): standard bfloat16
    else:
        llm = LLM(**common_kwargs)

    log.info("[%s] Model loaded.", model_key)
    return llm


# ---------------------------------------------------------------------------
# Build vLLM inputs per model family
# ---------------------------------------------------------------------------

def build_qwen_inputs(records: list[dict], model_key: str):
    """Build vLLM inputs for Qwen2-VL / Llama 4 Scout."""
    from transformers import AutoProcessor

    model_id = OPEN_MODELS[model_key]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    inputs = []
    for record in records:
        prompt_text = build_prompt(record)
        img = load_image(record.get("image_path", ""))

        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        entry = {"prompt": text}
        if img is not None:
            entry["multi_modal_data"] = {"image": img}
        inputs.append(entry)

    return inputs


def build_internvl_inputs(records: list[dict]):
    """Build vLLM inputs for InternVL2.5.

    InternVL2.5 requires the tokenizer chat template to produce the correct
    <|im_start|>...<|im_end|> conversation format. Using a bare '<image>\\n...'
    prompt causes the model to emit EOS immediately (0-token outputs).
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        OPEN_MODELS["internvl-8b"], trust_remote_code=True
    )

    inputs = []
    for record in records:
        prompt_text = build_prompt(record)
        img = load_image(record.get("image_path", ""))

        content = f"<image>\n{prompt_text}" if img is not None else prompt_text
        messages = [{"role": "user", "content": content}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        entry = {"prompt": text}
        if img is not None:
            entry["multi_modal_data"] = {"image": img}
        inputs.append(entry)

    return inputs


def build_phi_vision_inputs(records: list[dict]):
    """Build vLLM inputs for Phi-3.5-Vision.

    Phi3VProcessor does NOT support apply_chat_template.
    Use raw prompt format: <|user|>\\n<|image_1|>\\n{text}<|end|>\\n<|assistant|>\\n
    """
    inputs = []
    for record in records:
        prompt_text = build_prompt(record)
        img = load_image(record.get("image_path", ""))

        if img is not None:
            text = f"<|user|>\n<|image_1|>\n{prompt_text}<|end|>\n<|assistant|>\n"
            entry = {"prompt": text, "multi_modal_data": {"image": img}}
        else:
            text = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
            entry = {"prompt": text}
        inputs.append(entry)

    return inputs


def build_llava_inputs(records: list[dict], model_key: str):
    """Build vLLM inputs for LLaVA-OneVision / Gemma 3 (use processor chat template)."""
    from transformers import AutoProcessor

    model_id = OPEN_MODELS[model_key]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    inputs = []
    for record in records:
        prompt_text = build_prompt(record)
        img = load_image(record.get("image_path", ""))

        if img is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}]
        else:
            messages = [{"role": "user", "content": prompt_text}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        entry = {"prompt": text}
        if img is not None:
            entry["multi_modal_data"] = {"image": img}
        inputs.append(entry)

    return inputs


def build_inputs(model_key: str, records: list[dict]) -> list[dict]:
    # Qwen2-VL / Qwen3-VL family + Gemma3 + Aya Vision (all use AutoProcessor + apply_chat_template)
    if model_key in ("qwen-7b", "qwen-72b", "gemma3-27b", "qwen3-vl-30b", "aya-vision-8b"):
        return build_qwen_inputs(records, model_key)
    # InternVL2.5 family
    elif model_key == "internvl-8b":
        return build_internvl_inputs(records)
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference_vllm(
    model_key: str,
    lang: str,
    records: list[dict],
    llm,
    output_dir: Path,
) -> None:
    from vllm import SamplingParams

    out_file = output_dir / model_key / lang / "responses.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already done
    existing: dict[str, dict] = {}
    if out_file.exists():
        try:
            for r in json.load(open(out_file, encoding="utf-8")):
                existing[r["id"]] = r
        except Exception:
            pass
        log.info("[%s/%s] Resuming — %d already done.", model_key, lang, len(existing))

    to_process = [r for r in records if r["id"] not in existing]
    log.info("[%s/%s] Running %d records ...", model_key, lang, len(to_process))

    if not to_process:
        log.info("[%s/%s] Nothing to do.", model_key, lang)
        return

    sp_cfg = SAMPLING_PARAMS_CFG_COT if USE_COT else SAMPLING_PARAMS_CFG
    sampling_params = SamplingParams(**sp_cfg)

    # Build all inputs at once
    log.info("[%s/%s] Building inputs ...", model_key, lang)
    t_build = time.time()
    vllm_inputs = build_inputs(model_key, to_process)
    log.info("[%s/%s] Inputs built in %.1fs.", model_key, lang, time.time() - t_build)

    # Batch inference — vLLM handles all scheduling internally
    log.info("[%s/%s] Running batch inference ...", model_key, lang)
    t_infer = time.time()
    outputs = llm.generate(vllm_inputs, sampling_params)
    elapsed = time.time() - t_infer
    throughput = len(to_process) / elapsed
    log.info(
        "[%s/%s] Inference done: %.1fs | %.1f records/s",
        model_key, lang, elapsed, throughput,
    )

    # Collect results
    results = list(existing.values())
    ts = datetime.now(timezone.utc).isoformat()

    for record, output in zip(to_process, outputs):
        out0 = output.outputs[0] if output.outputs else None
        response_text = out0.text.strip() if out0 else "ERROR: no output"
        results.append({
            "id": record["id"],
            "source": record.get("source", ""),
            "category": record.get("category", ""),
            "lang": lang,
            "answer_type": record.get("answer_type", ""),
            "answer": record.get("answer", ""),
            "options": record.get("options", []),
            "question": record["question"],
            "model_response": response_text,
            "prompt_tokens": len(output.prompt_token_ids) if output.prompt_token_ids else None,
            "completion_tokens": len(out0.token_ids) if out0 else 0,
            "finish_reason": out0.finish_reason if out0 else None,
            "latency_sec": round(elapsed / len(to_process), 3),
            "timestamp": ts,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info("[%s/%s] Saved %d results → %s", model_key, lang, len(results), out_file)


# ---------------------------------------------------------------------------
# Batched multi-language inference (all langs in one generate call)
# ---------------------------------------------------------------------------

def run_inference_vllm_all_langs(
    model_key: str,
    langs: list[str],
    llm,
    output_dir: Path,
) -> None:
    """Batch ALL languages together in a single llm.generate() for max GPU throughput.

    Output: results/raw/{model_display_name}.jsonl  (one JSON line per inference call)
    """
    from vllm import SamplingParams

    sp_cfg = SAMPLING_PARAMS_CFG_COT if USE_COT else SAMPLING_PARAMS_CFG
    sampling_params = SamplingParams(**sp_cfg)
    model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)

    # JSONL output — one file per model, all languages combined
    suffix = "_cot" if USE_COT else ("_no_image" if NO_IMAGE else "")
    jsonl_path = output_dir / f"{model_display}{suffix}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-done (id, lang) pairs from existing JSONL
    done_keys: set[tuple[str, str]] = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
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

    # Build flat list of (lang, record) pairs to process
    flat_pairs: list[tuple[str, dict]] = []
    for lang in langs:
        records = load_records(lang)
        if not records:
            log.warning("[%s/%s] No records — skipping.", model_key, lang)
            continue
        for record in records:
            if (record["id"], lang) not in done_keys:
                flat_pairs.append((lang, record))

    if not flat_pairs:
        log.info("[%s] All languages complete — nothing to do.", model_key)
        return

    langs_present = sorted({lang for lang, _ in flat_pairs})
    total = len(flat_pairs)
    log.info("[%s] Batching %d inputs across %d languages ...", model_key, total, len(langs_present))

    t_build = time.time()
    flat_records = [r for _, r in flat_pairs]
    vllm_inputs = build_inputs(model_key, flat_records)
    log.info("[%s] Inputs built in %.1fs.", model_key, time.time() - t_build)

    log.info("[%s] Running batch inference (%d total) ...", model_key, total)
    t_infer = time.time()
    outputs = llm.generate(vllm_inputs, sampling_params)
    elapsed = time.time() - t_infer
    per_record_latency = round(elapsed / total, 3)
    log.info("[%s] Inference done: %.1fs | %.2f s/record", model_key, elapsed, per_record_latency)

    # Write results to JSONL
    ts = datetime.now(timezone.utc).isoformat()
    with open(jsonl_path, "a", encoding="utf-8") as f_out:
        for (lang, record), output in zip(flat_pairs, outputs):
            out0 = output.outputs[0] if output.outputs else None
            raw_response = out0.text.strip() if out0 else "ERROR: no output"

            answer_type = record.get("answer_type", "")
            options = record.get("options", [])
            ground_truth = str(record.get("answer", ""))

            extracted = extract_answer(raw_response, answer_type, options)
            correct = is_correct(extracted, ground_truth, answer_type)
            resp_lang = detect_response_language(raw_response)
            en_ratio = english_token_ratio(raw_response)
            lmeta = LANG_META.get(lang, {})
            en_rec = en_records.get(record["id"], {})

            entry = {
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
                    "response_length_tokens": len(out0.token_ids) if out0 else 0,
                    "response_length_chars": len(raw_response),
                    "finish_reason": out0.finish_reason if out0 else None,
                },
                "meta": {
                    "timestamp": ts,
                    "latency_seconds": per_record_latency,
                    "prompt_tokens": len(output.prompt_token_ids) if output.prompt_token_ids else None,
                    "temperature": sp_cfg.get("temperature", 0),
                    "max_tokens": sp_cfg.get("max_tokens", 512),
                    "prompt_template_version": "v1",
                },
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info("[%s] Saved %d results → %s", model_key, total, jsonl_path)


# ---------------------------------------------------------------------------
# HuggingFace results upload (private dataset)
# ---------------------------------------------------------------------------

HF_RESULTS_REPO = "Swastikr/multilingual-vlm-results"


def upload_model_results_to_hf(model_key: str, output_dir: Path, variant: str) -> None:
    """Upload results JSONL for one model to a private HF dataset repo."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set — skipping HF upload for %s.", model_key)
        return
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        try:
            create_repo(HF_RESULTS_REPO, repo_type="dataset", private=True, token=token)
        except Exception:
            pass  # Already exists

        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        suffix = "_cot" if variant == "cot" else ("_no_image" if variant == "no_image" else "")
        jsonl_path = output_dir / f"{model_display}{suffix}.jsonl"
        if not jsonl_path.exists():
            log.warning("No results JSONL for %s — skipping upload.", model_key)
            return

        api.upload_file(
            path_or_fileobj=str(jsonl_path),
            path_in_repo=f"results/{jsonl_path.name}",
            repo_id=HF_RESULTS_REPO,
            repo_type="dataset",
        )
        log.info("[%s] Results uploaded to HF: %s/results/%s", model_key, HF_RESULTS_REPO, jsonl_path.name)
    except Exception as exc:
        log.warning("[%s] HF upload failed (non-fatal): %s", model_key, exc)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(lang: str) -> list[dict]:
    if lang == "en":
        path = ROOT / "data" / "original" / "dataset.json"
    else:
        path = ROOT / "data" / "translated" / lang / "questions.json"
    if not path.exists():
        log.error("Data not found: %s — run translation first.", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global NO_IMAGE, USE_COT
    parser = argparse.ArgumentParser(description="Run open-source VLM inference via vLLM.")
    parser.add_argument(
        "--model", default="all",
        help="Model key(s): qwen-7b | qwen-72b | internvl-8b | all",
    )
    parser.add_argument(
        "--lang", default="all",
        help="Language code(s): en,hi,ta,te,bn,kn,mr or 'all'",
    )
    parser.add_argument(
        "--no-image", action="store_true",
        help="Text-only baseline: strip images from all inputs",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Chain-of-thought variant: append 'Think step by step' to prompt",
    )
    parser.add_argument(
        "--upload-hf", action="store_true",
        help="Upload results to private HF dataset after each model completes",
    )
    args = parser.parse_args()

    NO_IMAGE = args.no_image
    USE_COT = args.cot

    if args.model.strip().lower() == "all":
        target_models = list(OPEN_MODELS.keys())
    else:
        target_models = [m.strip() for m in args.model.split(",")]

    if args.lang.strip().lower() == "all":
        target_langs = LANGUAGES
    else:
        target_langs = [l.strip() for l in args.lang.split(",")]

    variant = "no_image" if NO_IMAGE else ("cot" if USE_COT else "standard")

    log.info("=" * 60)
    log.info("Step 04a — Open-Source VLM Inference (vLLM)")
    log.info("Models: %s", target_models)
    log.info("Languages: %s", target_langs)
    log.info("Variant: %s", variant)
    log.info("=" * 60)

    # All JSONL files go to results/raw/ directly (variant encoded in filename suffix)
    output_dir = ROOT / "results" / "raw"

    for model_key in target_models:
        if model_key not in OPEN_MODELS:
            log.warning("Unknown model '%s' — skipping.", model_key)
            continue

        log.info("-" * 50)
        log.info("Model: %s → %s", model_key, OPEN_MODELS[model_key])

        try:
            llm = load_vllm_model(model_key)
        except Exception as exc:
            log.error("Failed to load %s: %s — skipping.", model_key, exc)
            continue

        run_inference_vllm_all_langs(model_key, target_langs, llm, output_dir)

        # Upload results to HF private dataset
        if args.upload_hf:
            upload_model_results_to_hf(model_key, output_dir, variant)

        # Free GPU memory before next model
        log.info("Freeing GPU memory after %s ...", model_key)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        log.info("GPU memory freed.")

    log.info("=" * 60)
    log.info("All open-source inference complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
