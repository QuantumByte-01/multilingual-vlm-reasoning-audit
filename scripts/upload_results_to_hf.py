"""
Upload all raw inference results to HuggingFace once complete.
Waits for cot + llama4-maverick to finish, deduplicates, then uploads all JSONL files.

Usage:
    python scripts/upload_results_to_hf.py              # wait + upload
    python scripts/upload_results_to_hf.py --now        # upload immediately without waiting
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time, io
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

LANGS = ["en", "hi", "ta", "te", "bn", "kn", "mr"]
RESULTS_REPO = "Swastikr/multilingual-vlm-results"

ALL_RESULTS = [
    "qwen2.5-vl-7b.jsonl",
    "internvl2.5-8b.jsonl",
    "gpt-4o.jsonl",
    "gemma3-27b.jsonl",
    "qwen2.5-vl-32b.jsonl",
    "qwen3-vl-30b.jsonl",
    "llama4-maverick.jsonl",
    "qwen2.5-vl-7b-no_image.jsonl",
    "qwen2.5-vl-7b-cot.jsonl",
]

WAIT_FOR = ["qwen2.5-vl-7b-cot.jsonl", "llama4-maverick.jsonl"]


def count_records(path: Path):
    recs, langs = [], Counter()
    for l in path.open(encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(l.strip())
            recs.append(r)
            langs[r.get("language")] += 1
        except:
            pass
    return recs, langs


def dedup(path: Path):
    seen, good = set(), []
    for l in path.open(encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(l.strip())
            key = str(r.get("id", "")) + str(r.get("language", ""))
            if key not in seen:
                seen.add(key)
                good.append(json.dumps(r, ensure_ascii=False))
        except:
            pass
    path.write_text("\n".join(good) + "\n", encoding="utf-8")
    return len(good)


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    _, langs = count_records(path)
    return all(langs.get(l, 0) >= 980 for l in LANGS)


def wait_for_completion():
    log.info("Waiting for inference jobs to complete...")
    while True:
        pending = []
        for fname in WAIT_FOR:
            p = ROOT / "results" / "raw" / fname
            if not is_complete(p):
                _, langs = count_records(p) if p.exists() else ([], Counter())
                missing = {l: 980 - langs.get(l, 0) for l in LANGS if langs.get(l, 0) < 980}
                pending.append(f"{fname} missing={missing}")
        if not pending:
            log.info("All jobs complete!")
            break
        log.info("Still waiting: %s", " | ".join(pending))
        time.sleep(60)


def upload_results(token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    raw_dir = ROOT / "results" / "raw"
    uploaded, skipped = [], []

    for fname in ALL_RESULTS:
        p = raw_dir / fname
        if not p.exists():
            log.warning("SKIP %s — file not found", fname)
            skipped.append(fname)
            continue

        # Dedup before upload
        n_before = sum(1 for l in p.open(encoding="utf-8", errors="ignore") if l.strip())
        n_after = dedup(p)
        if n_before != n_after:
            log.info("Deduped %s: %d -> %d records", fname, n_before, n_after)

        _, langs = count_records(p)
        missing = {l: 980 - langs.get(l, 0) for l in LANGS if langs.get(l, 0) < 980}
        if missing:
            log.warning("INCOMPLETE %s — missing=%s — uploading anyway", fname, missing)

        log.info("Uploading %s (%d records)...", fname, n_after)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"results/raw/{fname}",
            repo_id=RESULTS_REPO,
            repo_type="dataset",
        )
        uploaded.append(fname)
        log.info("  -> uploaded %s", fname)

    log.info("=" * 60)
    log.info("Upload complete: %d uploaded, %d skipped", len(uploaded), len(skipped))
    log.info("View at: https://huggingface.co/datasets/%s", RESULTS_REPO)
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Upload immediately without waiting")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set in .env")
        sys.exit(1)

    if not args.now:
        wait_for_completion()

    # Final dedup pass on all files
    log.info("Running final dedup on all result files...")
    for fname in ALL_RESULTS:
        p = ROOT / "results" / "raw" / fname
        if p.exists():
            n = dedup(p)
            log.info("  %s: %d records", fname, n)

    upload_results(token)


if __name__ == "__main__":
    main()
