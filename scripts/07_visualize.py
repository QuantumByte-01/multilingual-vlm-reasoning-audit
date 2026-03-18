"""
Script 07 — Generate Paper Figures
====================================
Produces all 4 figures described in the paper:

  Figure 1 — Heatmap: accuracy by model × language
  Figure 2 — Grouped bar chart: average accuracy drop per language
  Figure 3 — Radar chart: reasoning type breakdown per language
  Figure 4 — Qualitative examples (saved as JSON for manual LaTeX rendering)

Output:
  figures/fig1_heatmap.pdf
  figures/fig2_accuracy_drop.pdf
  figures/fig3_radar.pdf
  figures/fig4_qualitative.json

Usage:
  python scripts/07_visualize.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

LANGUAGE_ORDER = ["English", "Hindi", "Marathi", "Bengali", "Tamil", "Telugu", "Kannada"]
LANG_CODE_ORDER = ["en", "hi", "mr", "bn", "ta", "te", "kn"]
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
}

# Publication-quality style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})


# ---------------------------------------------------------------------------
# Figure 1 — Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame) -> None:
    """Figure 1: accuracy heatmap, models × languages."""
    # Sort models by English accuracy descending (matches Table 1 ordering)
    MODEL_ORDER = [
        "Qwen3-VL-30B", "Llama-4-Maverick", "GPT-4o", "Gemma 3-27B",
        "Qwen2.5-VL-7B", "Qwen2.5-VL-32B", "InternVL2.5-8B", "Aya-Vision-8B",
    ]
    lang_cols = [l for l in LANGUAGE_ORDER if l in df["language"].values]
    pivot = (
        df.pivot(index="model", columns="language", values="accuracy")
        .reindex(columns=lang_cols)
    )
    # Apply model order (only rows that exist)
    ordered = [m for m in MODEL_ORDER if m in pivot.index]
    pivot = pivot.reindex(ordered)

    fig, ax = plt.subplots(figsize=(len(lang_cols) * 1.4, max(3, len(pivot) * 1.0) + 1.0))

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=30,
        vmax=85,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        annot_kws={"size": 9},
    )

    ax.set_title("Visual Reasoning Accuracy (%) by Model and Language", pad=14)
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # Vertical separator between English and Indian languages
    ax.axvline(x=1, color="black", linewidth=2, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_heatmap.pdf"
    plt.savefig(out)
    plt.close()
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 2 — Accuracy drop bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_drop(df: pd.DataFrame) -> None:
    """Figure 2: average accuracy drop per language across all models."""
    df_drop = df[df["lang_code"] != "en"].copy()
    avg_drop = (
        df_drop.groupby(["lang_code", "language"])["drop_abs"]
        .mean()
        .reset_index()
    )

    # Sort by ascending drop (Hindi first, Kannada last)
    lang_order = [l for l in LANG_CODE_ORDER if l != "en"]
    avg_drop["lang_code"] = pd.Categorical(avg_drop["lang_code"], categories=lang_order, ordered=True)
    avg_drop = avg_drop.sort_values("lang_code")

    # Color: Indo-Aryan = blue family, Dravidian = red family
    colors = []
    for lc in avg_drop["lang_code"]:
        if lc in {"hi", "mr", "bn"}:
            colors.append("#4472C4")
        else:
            colors.append("#C00000")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        avg_drop["language"],
        avg_drop["drop_abs"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, avg_drop["drop_abs"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title("Average Accuracy Drop from English (Across All Models)", pad=12)
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy Drop (percentage points)")
    ax.set_ylim(0, avg_drop["drop_abs"].max() + 8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4472C4", label="Indo-Aryan"),
        Patch(facecolor="#C00000", label="Dravidian"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_accuracy_drop.pdf"
    plt.savefig(out)
    plt.close()
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 3 — Radar chart
# ---------------------------------------------------------------------------

def plot_radar(df_cat: pd.DataFrame, models_to_show: list[str] | None = None) -> None:
    """Figure 3: reasoning type breakdown per language (radar chart)."""
    categories = sorted(df_cat["category"].unique())
    n_cat = len(categories)
    if n_cat < 3:
        log.warning("Too few categories for radar chart — skipping Figure 3.")
        return

    # Average across models
    avg = (
        df_cat.groupby(["language", "category"])["accuracy"]
        .mean()
        .reset_index()
    )

    # Pick languages to show (English + 2-3 interesting ones)
    langs_to_show = ["English", "Hindi", "Tamil", "Kannada"]
    langs_to_show = [l for l in langs_to_show if l in avg["language"].values]

    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]  # close the radar

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for i, lang in enumerate(langs_to_show):
        lang_data = avg[avg["language"] == lang]
        values = [
            lang_data[lang_data["category"] == cat]["accuracy"].values[0]
            if cat in lang_data["category"].values else 0
            for cat in categories
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=lang)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    CATEGORY_DISPLAY = {"math": "MathVista", "science": "ScienceQA", "stem": "MMMU"}
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CATEGORY_DISPLAY.get(c, c.capitalize()) for c in categories], size=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=8)
    ax.set_title("Reasoning Type Performance by Language", pad=20, size=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_radar.pdf"
    plt.savefig(out)
    plt.close()
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 4 — Qualitative examples (exported as JSON for LaTeX)
# ---------------------------------------------------------------------------

def export_qualitative(all_results: dict) -> None:
    """
    Find examples where English is correct but Hindi/Tamil is wrong.
    Saves to figures/fig4_qualitative.json for manual LaTeX formatting.
    """
    examples: list[dict] = []
    target_pairs = [("en", "hi"), ("en", "ta"), ("en", "kn")]

    for model_name, records in all_results.items():
        en_recs = records.get("en", [])
        en_by_id = {r["id"]: r for r in en_recs}

        for en_lang, tgt_lang in target_pairs:
            tgt_recs = records.get(tgt_lang, [])
            for r in tgt_recs:
                en_r = en_by_id.get(r["id"])
                if not en_r:
                    continue
                if en_r["correct"] and not r["correct"] and not r.get("extraction_failed"):
                    examples.append({
                        "model": model_name,
                        "id": r["id"],
                        "source": r.get("source", ""),
                        "category": r.get("category", ""),
                        "english_question": en_r.get("question", ""),
                        "target_lang": tgt_lang,
                        "target_question": r.get("question", ""),
                        "ground_truth": r.get("answer", ""),
                        "english_response": en_r.get("model_response", ""),
                        "target_response": r.get("model_response", ""),
                        "image_path": "",
                    })
                    if len(examples) >= 10:
                        break
            if len(examples) >= 10:
                break
        if len(examples) >= 10:
            break

    out = FIGURES_DIR / "fig4_qualitative.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(examples[:5], f, ensure_ascii=False, indent=2)
    log.info("Saved %d qualitative examples to %s", len(examples[:5]), out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Step 07 — Generate Figures")
    log.info("=" * 60)

    processed_dir = ROOT / "results" / "processed"
    summary_file = processed_dir / "summary.json"
    main_table_file = processed_dir / "main_table.csv"
    cat_table_file = processed_dir / "category_table.csv"

    if not main_table_file.exists():
        log.error("main_table.csv not found. Run script 06 first.")
        sys.exit(1)

    df_main = pd.read_csv(main_table_file)
    df_cat = pd.read_csv(cat_table_file) if cat_table_file.exists() else pd.DataFrame()

    log.info("Loaded main_table: %d rows | category_table: %d rows", len(df_main), len(df_cat))

    ABLATIONS = {"qwen2.5-vl-7b-cot", "qwen2.5-vl-7b-no_image"}
    MODEL_DISPLAY = {
        "aya-vision-8b":       "Aya-Vision-8B",
        "gemma3-27b":          "Gemma 3-27B",
        "gpt-4o":              "GPT-4o",
        "internvl2.5-8b":      "InternVL2.5-8B",
        "llama4-maverick":     "Llama-4-Maverick",
        "qwen2.5-vl-32b":      "Qwen2.5-VL-32B",
        "qwen2.5-vl-7b":       "Qwen2.5-VL-7B",
        "qwen3-vl-30b":        "Qwen3-VL-30B",
    }
    df_models = df_main[~df_main["model"].isin(ABLATIONS)].copy()
    df_models["model"] = df_models["model"].map(MODEL_DISPLAY).fillna(df_models["model"])

    plot_heatmap(df_models)
    plot_accuracy_drop(df_models)

    if not df_cat.empty:
        plot_radar(df_cat)

    # Qualitative examples — load from extracted results
    ext_results: dict[str, dict[str, list[dict]]] = {}
    for model_dir in sorted(processed_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.endswith(".csv") or model_dir.name.endswith(".json"):
            continue
        ext_results[model_dir.name] = {}
        for lang_dir in sorted(model_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            ext_file = lang_dir / "extracted.json"
            if ext_file.exists():
                with open(ext_file, encoding="utf-8") as f:
                    ext_results[model_dir.name][lang_dir.name] = json.load(f)

    if ext_results:
        export_qualitative(ext_results)

    log.info("=" * 60)
    log.info("All figures generated in %s/", FIGURES_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
