# Do Multilingual VLMs Reason Equally?
### A Cross-Lingual Visual Reasoning Audit for Indian Languages

**Paper:** [arXiv:2603.26742](https://arxiv.org/abs/2603.26742)
**Dataset:** [HuggingFace](https://huggingface.co/datasets/Swastikr/multilingual-vlm-reasoning)

## Key Findings

| Finding | Result |
|---------|--------|
| Average accuracy drop (English → Indian languages) | 9.8–25 pp |
| Dravidian vs Indo-Aryan gap | Up to 13.2 pp |
| Chain-of-thought in Indian languages | Hurts (up to −14.4 pp) |
| Models tested | 8 |
| Total inference records | 68,600 |

## Languages
English, Hindi, Tamil, Telugu, Bengali, Kannada, Marathi

## Models Evaluated
GPT-4o, Gemma 3-27B, Llama-4-Maverick, Qwen3-VL-30B,  
Qwen2.5-VL-32B, Qwen2.5-VL-7B, InternVL2.5-8B, Aya-Vision-8B

## Datasets
- MathVista (400 questions)
- ScienceQA (400 questions)
- MMMU (180 questions)

## Quick Start
```bash
pip install -r requirements.txt

# Translate questions
python scripts/02_translate.py

# Run inference
python scripts/04_run_inference_api.py
python scripts/04_run_inference_open.py

# Evaluate
python scripts/06_evaluate.py

# Generate figures
python scripts/07_visualize.py
```

## Repository Structure
```
├── data/
│   ├── original/          # English source questions
│   └── translated/        # Translated questions (hi/ta/te/bn/kn/mr)
├── scripts/               # Full pipeline: translate → infer → evaluate → visualize
├── results/
│   └── processed/         # Summary CSV and metrics
├── figures/               # Paper figures (fig1–fig10)
└── paper/
    ├── main.tex
    ├── references.bib
    └── paper.pdf
```

## Citation
```bibtex
@article{swastik2026multilingual,
  title={Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages},
  author={Swastik R},
  journal={arXiv preprint arXiv:2603.26742},
  year={2026}
}
```

## License
MIT
