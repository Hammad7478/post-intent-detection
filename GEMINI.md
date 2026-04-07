# Project Memory: Post Intent Detection

## 📅 Last Updated: April 6, 2026 (Final Project Completion)
**Current Status:** PROJECT COMPLETE. All models (A, B, C, D) are implemented, evaluated, and documented. Final report and submission package are finalized.

## 🎯 Model Performance Summary (on TEST set)

| Model | Architecture | Accuracy | Weighted F1 | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Majority Class (Advice) | 31.10% | 14.75% | 11.86% | Done |
| **Model A** | TF-IDF (Word) + LogReg | 73.78% | 72.66% | 69.49% | Done |
| **Model B** | Hybrid TF-IDF (Word+Char) | 75.30% | 74.33% | 70.92% | Done |
| **Model C** | DistilBERT (Fine-tuned) | 75.91% | 75.84% | 74.02% | Done |
| **Model D** | RoBERTa-base (Proposed) | **82.32%** | **82.49%** | **81.42%** | **Done** |

## ✅ Final Accomplishments
1.  **Model D Optimization**: Transitioned from DeBERTa to **RoBERTa-base** to resolve security/compatibility issues. Enabled **GPU training** (RTX 3060) and achieved state-of-the-art results (82.32% accuracy).
2.  **Imbalance Mitigation**: Implemented a `CustomTrainer` with **Class Weights** ($W_{Other} \approx 1.68$) to significantly improve performance on the minority "OTHER" class.
3.  **High-Detail Final Report**: Authored a comprehensive Final Report in LaTeX (`Final Project TeX/main.tex`).
    - Matched the high-detail standards of a previous 100%-graded project.
    - Included architectural ablation studies, qualitative error analysis (case study `t3_1qdzso8`), and competitor comparisons.
4.  **Standalone LaTeX Implementation**: Re-engineered the report into a self-contained LaTeX file to eliminate external dependencies (`acl.sty`, `custom.bib`) and ensure 100% compilation success.
5.  **Final Submission Package**: Generated `submission.json` (Test) and `submission_val.json` (Validation) and created the final archive: `Group27_Final_Submission.zip`.

## 🏁 Final Deliverables
- **Best Model**: Model D (RoBERTa-base)
- **Report**: `Final Project TeX/main.tex` (Ready for PDF generation)
- **Submission**: `Group27_Final_Submission.zip` (Ready for Avenue to Learn)

## 📁 Key Directories
- `models/`: All implementation scripts (including the final `model_d_roberta.py`).
- `models/artifacts_d_roberta/`: Saved weights and artifacts for the best model.
- `data/data_splits/`: Standardized train/val/test splits.
- `Final Project TeX/`: Self-contained LaTeX source for the final report.
