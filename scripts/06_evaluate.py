"""
Script 06 — Evaluate Results and Compute Metrics
=================================================
Aggregates extracted answers and computes:
  - Accuracy per model × language
  - Accuracy drop (Δ) vs. English baseline
  - Per-category accuracy (math / science / stem)
  - Language family comparison (Indo-Aryan vs. Dravidian)
  - Cross-lingual consistency (% same answer across all languages)
  - Refusal rate, extraction failure rate
  - McNemar's test (pairwise significance vs. English)
  - 95% bootstrap confidence intervals

Output:
  results/processed/summary.json     — full metrics table
  results/processed/main_table.csv   — Table 1 (accuracy per model × language)
  results/processed/category_table.csv — Table 2 (per-category breakdown)

Usage:
  python scripts/06_evaluate.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LANGUAGES = ["en", "hi", "mr", "bn", "ta", "te", "kn"]
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
}
INDO_ARYAN = {"hi", "mr", "bn"}
DRAVIDIAN = {"ta", "te", "kn"}

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(correct: list[bool], n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Return (lower_95, upper_95) bootstrap CI for accuracy."""
    rng = np.random.default_rng(seed)
    arr = np.array(correct, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(en_correct: list[bool], lang_correct: list[bool]) -> float:
    """McNemar's test p-value comparing English vs. target-language correctness."""
    # b = english correct, lang wrong; c = english wrong, lang correct
    b = sum(1 for e, l in zip(en_correct, lang_correct) if e and not l)
    c = sum(1 for e, l in zip(en_correct, lang_correct) if not e and l)
    if b + c == 0:
        return 1.0
    # Use exact binomial test for small samples
    if b + c < 25:
        result = stats.binomtest(b, b + c, 0.5, alternative="two-sided")
        return float(result.pvalue)
    # Chi-squared approximation
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - stats.chi2.cdf(chi2, df=1))


# ---------------------------------------------------------------------------
# Load all extracted results
# ---------------------------------------------------------------------------

def load_all_results(processed_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """Returns {(model_name, lang): [extracted_records]}"""
    results: dict[tuple[str, str], list[dict]] = {}
    for model_dir in sorted(processed_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "summary.json":
            continue
        for lang_dir in sorted(model_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            ext_file = lang_dir / "extracted.json"
            if ext_file.exists():
                with open(ext_file, encoding="utf-8") as f:
                    results[(model_dir.name, lang_dir.name)] = json.load(f)
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Step 06 — Evaluate Results")
    log.info("=" * 60)

    processed_dir = ROOT / "results" / "processed"
    all_results = load_all_results(processed_dir)

    if not all_results:
        log.error("No extracted results found. Run script 05 first.")
        sys.exit(1)

    models = sorted({m for m, _ in all_results.keys()})
    log.info("Models found: %s", models)

    summary: dict = {"models": {}}
    rows_main: list[dict] = []
    rows_category: list[dict] = []

    for model in models:
        model_data: dict = {}

        # Per-language metrics
        en_records = all_results.get((model, "en"), [])
        en_by_id = {r["id"]: r for r in en_records}

        for lang in LANGUAGES:
            records = all_results.get((model, lang), [])
            if not records:
                continue

            correct = [bool(r["correct"]) for r in records]
            n_total = len(correct)
            n_failed = sum(1 for r in records if r.get("extraction_failed"))
            n_correct = sum(correct)
            acc = n_correct / n_total * 100 if n_total else 0.0
            ci_low, ci_high = bootstrap_accuracy_ci(correct)

            refusal_rate = sum(
                1 for r in records
                if "i don't" in r.get("model_response", "").lower()
                or "cannot" in r.get("model_response", "").lower()
            ) / n_total if n_total else 0.0

            # Accuracy drop vs. English
            if lang != "en" and en_records:
                en_correct_aligned = []
                lang_correct_aligned = []
                for r in records:
                    en_r = en_by_id.get(r["id"])
                    if en_r:
                        en_correct_aligned.append(en_r["correct"])
                        lang_correct_aligned.append(r["correct"])

                en_acc = np.mean(en_correct_aligned) * 100
                lang_acc = np.mean(lang_correct_aligned) * 100
                delta = en_acc - lang_acc
                rel_drop = (delta / en_acc * 100) if en_acc > 0 else 0.0
                p_value = mcnemar_test(en_correct_aligned, lang_correct_aligned)
            else:
                delta = 0.0
                rel_drop = 0.0
                p_value = 1.0

            model_data[lang] = {
                "n_total": n_total,
                "n_correct": n_correct,
                "n_failed_extract": n_failed,
                "accuracy": round(acc, 2),
                "ci_95_low": round(ci_low * 100, 2),
                "ci_95_high": round(ci_high * 100, 2),
                "accuracy_drop_abs": round(delta, 2),
                "accuracy_drop_rel_pct": round(rel_drop, 2),
                "mcnemar_p": round(p_value, 4),
                "significant": p_value < 0.05,
                "refusal_rate": round(refusal_rate, 4),
            }

            rows_main.append({
                "model": model,
                "language": LANGUAGE_NAMES.get(lang, lang),
                "lang_code": lang,
                "accuracy": round(acc, 2),
                "ci_95": f"[{round(ci_low*100,1)}, {round(ci_high*100,1)}]",
                "drop_abs": round(delta, 2),
                "drop_rel_pct": round(rel_drop, 2),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
                "n": n_total,
            })

            # Per-category breakdown
            for category in ("math", "science", "stem"):
                cat_records = [r for r in records if r.get("category") == category]
                if not cat_records:
                    continue
                cat_correct = [r["correct"] for r in cat_records]
                cat_acc = sum(cat_correct) / len(cat_correct) * 100

                rows_category.append({
                    "model": model,
                    "language": LANGUAGE_NAMES.get(lang, lang),
                    "lang_code": lang,
                    "category": category,
                    "accuracy": round(cat_acc, 2),
                    "n": len(cat_records),
                })

        # Language family analysis
        ia_accs = [model_data[l]["accuracy"] for l in INDO_ARYAN if l in model_data]
        dr_accs = [model_data[l]["accuracy"] for l in DRAVIDIAN if l in model_data]
        if ia_accs and dr_accs:
            t_stat, t_p = stats.ttest_ind(ia_accs, dr_accs)
            model_data["family_analysis"] = {
                "indo_aryan_mean": round(float(np.mean(ia_accs)), 2),
                "dravidian_mean": round(float(np.mean(dr_accs)), 2),
                "gap": round(float(np.mean(ia_accs) - np.mean(dr_accs)), 2),
                "t_stat": round(float(t_stat), 4),
                "t_p": round(float(t_p), 4),
            }

        # Cross-lingual consistency
        if en_records:
            all_lang_by_id: dict[str, list[str | None]] = {}
            for lang in LANGUAGES:
                recs = all_results.get((model, lang), [])
                for r in recs:
                    all_lang_by_id.setdefault(r["id"], []).append(r.get("predicted"))

            consistent = sum(
                1 for preds in all_lang_by_id.values()
                if preds and len(set(str(p) for p in preds)) == 1
            )
            model_data["consistency_pct"] = round(
                consistent / len(all_lang_by_id) * 100 if all_lang_by_id else 0.0, 2
            )

        summary["models"][model] = model_data

    # Save summary JSON
    with open(processed_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Saved summary.json")

    # Save main table CSV
    df_main = pd.DataFrame(rows_main)
    df_main.to_csv(processed_dir / "main_table.csv", index=False)
    log.info("Saved main_table.csv")

    # Save category table CSV
    df_cat = pd.DataFrame(rows_category)
    df_cat.to_csv(processed_dir / "category_table.csv", index=False)
    log.info("Saved category_table.csv")

    # Print summary to console
    log.info("")
    log.info("=== ACCURACY TABLE (%) ===")
    if not df_main.empty:
        pivot = df_main.pivot(index="model", columns="language", values="accuracy")
        log.info("\n%s", pivot.to_string())

    log.info("")
    log.info("=== ACCURACY DROP (abs %%) vs. English ===")
    if not df_main.empty:
        drop_pivot = df_main[df_main.lang_code != "en"].pivot(
            index="model", columns="language", values="drop_abs"
        )
        log.info("\n%s", drop_pivot.to_string())

    log.info("=" * 60)
    log.info("Evaluation complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
