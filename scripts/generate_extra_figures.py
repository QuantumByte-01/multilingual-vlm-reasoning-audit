"""Generate additional analysis figures (fig5-fig10)."""
import json, sys, io, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

LANGS = ['en','hi','ta','te','bn','kn','mr']
LANG_NAMES = {'en':'English','hi':'Hindi','ta':'Tamil','te':'Telugu','bn':'Bengali','kn':'Kannada','mr':'Marathi'}
MAIN_MODELS = ['gpt-4o','gemma3-27b','llama4-maverick','qwen3-vl-30b','qwen2.5-vl-32b','qwen2.5-vl-7b','internvl2.5-8b','aya-vision-8b']
MODEL_SHORT = {
    'gpt-4o':'GPT-4o','gemma3-27b':'Gemma3-27B','llama4-maverick':'Llama4-Mav',
    'qwen3-vl-30b':'Qwen3-30B','qwen2.5-vl-32b':'Qwen2.5-32B','qwen2.5-vl-7b':'Qwen2.5-7B',
    'internvl2.5-8b':'InternVL-8B','aya-vision-8b':'Aya-V-8B'
}
COLORS = ['#e41a1c','#377eb8','#ff7f00','#4daf4a','#984ea3','#a65628','#f781bf','#999999']
SCRIPT_RANGES = {
    'hi':(0x0900,0x097F),'mr':(0x0900,0x097F),
    'ta':(0x0B80,0x0BFF),'te':(0x0C00,0x0C7F),
    'bn':(0x0980,0x09FF),'kn':(0x0C80,0x0CFF),
}

ROOT = Path(__file__).resolve().parent.parent
MODELS = {}
for f in sorted((ROOT / 'results' / 'raw').glob('*.jsonl')):
    if 'gemini' in f.stem: continue
    recs = [json.loads(l) for l in f.open(encoding='utf-8', errors='ignore') if l.strip()]
    MODELS[f.stem] = recs

FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def acc(recs):
    if not recs: return 0.0
    return sum(bool(r['output'].get('correct')) for r in recs) / len(recs) * 100


# ── Fig 5: Per-dataset accuracy grouped bars ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, ds, dsl in zip(axes, ['mathvista','scienceqa','mmmu'], ['MathVista','ScienceQA','MMMU']):
    en_accs, in_accs = [], []
    for name in MAIN_MODELS:
        recs = MODELS[name]
        ds_recs = [r for r in recs if r.get('dataset') == ds]
        en_accs.append(acc([r for r in ds_recs if r['language'] == 'en']))
        in_accs.append(acc([r for r in ds_recs if r['language'] != 'en']))
    x = np.arange(len(MAIN_MODELS)); w = 0.35
    ax.bar(x - w/2, en_accs, w, label='English', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, in_accs, w, label='Indian avg', color='tomato', alpha=0.85)
    ax.set_title(dsl, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MAIN_MODELS], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
plt.suptitle('Per-Dataset Accuracy: English vs. Average Indian Languages', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_per_dataset.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig5_per_dataset.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig5_per_dataset')


# ── Fig 6: Language confusion heatmap ─────────────────────────────────────────
lang_order = ['hi','mr','bn','ta','te','kn']
conf_matrix = np.zeros((len(MAIN_MODELS), len(lang_order)))
for mi, name in enumerate(MAIN_MODELS):
    recs = MODELS[name]
    for li, lang in enumerate(lang_order):
        lo, hi_ = SCRIPT_RANGES[lang]
        lr = [r for r in recs if r.get('language') == lang]
        verbose = [r for r in lr if len(str(r['output'].get('raw_response', ''))) > 15]
        if not verbose: continue
        confused = sum(1 for r in verbose if not any(lo <= ord(c) <= hi_ for c in str(r['output']['raw_response'])))
        conf_matrix[mi, li] = confused / len(verbose) * 100

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(conf_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=80)
ax.set_xticks(range(len(lang_order)))
ax.set_xticklabels([LANG_NAMES[l] for l in lang_order], fontsize=10)
ax.set_yticks(range(len(MAIN_MODELS)))
ax.set_yticklabels([MODEL_SHORT[m] for m in MAIN_MODELS], fontsize=10)
for i in range(len(MAIN_MODELS)):
    for j in range(len(lang_order)):
        v = conf_matrix[i, j]
        ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=9,
                color='white' if v > 50 else 'black')
plt.colorbar(im, ax=ax, label='Language Confusion (%)')
ax.set_title('Language Confusion Rate: % Responses Missing Target Script', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_confusion_heatmap.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig6_confusion_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig6_confusion_heatmap')


# ── Fig 7: Cross-lingual consistency ─────────────────────────────────────────
consistencies = []
for name in MAIN_MODELS:
    recs = MODELS[name]
    by_qid = defaultdict(dict)
    for r in recs:
        qid = r.get('question_id', r.get('id', ''))
        by_qid[qid][r['language']] = r['output'].get('extracted_answer')
    consis = []
    for qid, la in by_qid.items():
        answers = [v for v in la.values() if v is not None]
        if len(answers) < 4: continue
        mc = max(set(answers), key=answers.count)
        consis.append(answers.count(mc) / len(answers))
    consistencies.append(sum(consis) / len(consis) * 100 if consis else 0)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MAIN_MODELS))
bars = ax.bar(x, consistencies, color=COLORS, alpha=0.85, edgecolor='white')
ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Random guess (4-choice MCQ, 25%)')
for bar, val in zip(bars, consistencies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_SHORT[m] for m in MAIN_MODELS], rotation=30, ha='right')
ax.set_ylabel('Cross-Lingual Consistency (%)'); ax.set_ylim(0, 100)
ax.set_title('Cross-Lingual Consistency: % Questions with Same Answer Across All 7 Languages', fontsize=11, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_consistency.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig7_consistency.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig7_consistency')


# ── Fig 8: Ablation comparison ────────────────────────────────────────────────
ablation_models = ['qwen2.5-vl-7b', 'qwen2.5-vl-7b-no_image', 'qwen2.5-vl-7b-cot']
ablation_labels = ['Standard', 'No-Image\n(text-only)', 'Chain-of-Thought']
ablation_colors = ['#2166ac', '#d1e5f0', '#ef8a62']

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(LANGS)); w = 0.25
for i, (name, label, color) in enumerate(zip(ablation_models, ablation_labels, ablation_colors)):
    recs = MODELS[name]
    accs = [acc([r for r in recs if r['language'] == l]) for l in LANGS]
    ax.bar(x + (i - 1) * w, accs, w, label=label, color=color, alpha=0.9, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([LANG_NAMES[l] for l in LANGS])
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 70)
ax.set_title('Qwen2.5-VL-7B Ablation: Standard vs. No-Image vs. Chain-of-Thought', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig8_ablation.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig8_ablation.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig8_ablation')


# ── Fig 9: Indo-Aryan vs Dravidian scatter ────────────────────────────────────
ia_langs = ['hi', 'mr', 'bn']
dr_langs = ['ta', 'te', 'kn']

LABEL_OFFSETS = {
    'internvl2.5-8b': (6, 4),
    'qwen2.5-vl-7b':  (6, -12),
    'aya-vision-8b':  (-60, 6),
}
fig, ax = plt.subplots(figsize=(8, 6))
for i, name in enumerate(MAIN_MODELS):
    recs = MODELS[name]
    en = acc([r for r in recs if r['language'] == 'en'])
    ia = np.mean([acc([r for r in recs if r['language'] == l]) for l in ia_langs])
    dr = np.mean([acc([r for r in recs if r['language'] == l]) for l in dr_langs])
    ax.scatter(en - ia, en - dr, s=120, color=COLORS[i], zorder=3, label=MODEL_SHORT[name])
    offset = LABEL_OFFSETS.get(name, (6, 4))
    ax.annotate(MODEL_SHORT[name], (en - ia, en - dr), textcoords='offset points', xytext=offset, fontsize=8)
lim = [0, 33]
ax.plot(lim, lim, 'k--', alpha=0.4, label='Dravidian = Indo-Aryan')
ax.set_xlabel('Accuracy Drop: Indo-Aryan (hi/mr/bn) pp', fontsize=11)
ax.set_ylabel('Accuracy Drop: Dravidian (ta/te/kn) pp', fontsize=11)
ax.set_title('Indo-Aryan vs. Dravidian Accuracy Drop\n(above diagonal = Dravidian harder)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 33); ax.set_ylim(0, 33)
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig9_family_scatter.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig9_family_scatter.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig9_family_scatter')


# ── Fig 10: English token leak ────────────────────────────────────────────────
leak_langs = ['hi', 'ta', 'te', 'bn', 'kn', 'mr']
leak_colors = plt.cm.Set2(np.linspace(0, 1, len(leak_langs)))

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(MAIN_MODELS)); w = 0.13
for i, lang in enumerate(leak_langs):
    rates = []
    for name in MAIN_MODELS:
        recs = MODELS[name]
        lr = [r['output'].get('english_token_ratio') or 0 for r in recs if r.get('language') == lang]
        rates.append(sum(lr) / len(lr) * 100 if lr else 0)
    ax.bar(x + (i - 2.5) * w, rates, w, label=LANG_NAMES[lang], color=leak_colors[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_SHORT[m] for m in MAIN_MODELS], rotation=30, ha='right')
ax.set_ylabel('English Token Ratio (%)')
ax.set_title('English Token Leak Rate in Non-English Responses', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig10_english_leak.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig10_english_leak.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig10_english_leak')


# ── Fig 11: Response length ratio ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(MAIN_MODELS)); w = 0.11
for i, lang in enumerate(leak_langs):
    ratios = []
    for name in MAIN_MODELS:
        recs = MODELS[name]
        en_len = np.mean([r['output'].get('response_length_chars') or 0 for r in recs if r['language'] == 'en']) or 1
        lang_len = np.mean([r['output'].get('response_length_chars') or 0 for r in recs if r['language'] == lang]) or 0
        ratios.append(lang_len / en_len)
    ax.bar(x + (i - 2.5) * w, ratios, w, label=LANG_NAMES[lang], color=leak_colors[i], alpha=0.85)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Same as English')
ax.set_xticks(x)
ax.set_xticklabels([MODEL_SHORT[m] for m in MAIN_MODELS], rotation=30, ha='right')
ax.set_ylabel('Response Length Ratio (lang / English)')
ax.set_title('Response Length Ratio per Language vs. English', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=4); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig11_response_length.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig11_response_length.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fig11_response_length')

print('\nAll extra figures saved.')
