---
name: epilepsy-project-context
description: >-
  Project context for Zhong Ziwen's BBC6521 Final Year Project: AI-Assisted
  Epilepsy Detection using EEG Signals (BUPT-QMUL Joint Programme 2025/26).
  Use when writing the final report LaTeX, modifying Bonn model code, analysing
  experiment results, preparing viva slides, or any task related to this project.
  Always read this skill first before making changes.
---

# AI-Assisted Epilepsy Detection — Project Context

This is **Zhong Ziwen's** BBC6521 Final Year Project.
Treat it as your own — be thorough, rigorous, and aligned with academic standards.

## 1. Mandatory First Step

Before any work on this project, **read the docs directory** to understand history and requirements:

```bash
# Read in this priority order:
Bonn/docs/cursor_deep_learning_model_with_bonn_da.md  # Full agent collaboration history
Bonn/docs/mid_term_report.md                          # Current mid-term report (latest results)
Bonn/docs/Zhong Ziwen_2022213356_Spec.pdf             # Project specification & mid-term targets
Bonn/docs/JP JEI_Project_Student_Handbook_2025-26.pdf  # School rules, deadlines, marking criteria
Bonn/docs/Report_GenAIRef_Risk_Environment_Viva_2025-26.pdf  # GenAI acknowledgement & viva format
Bonn/docs/Zhong Ziwen_2022213356_EarlyTerm.pdf        # Early-term progress report
Bonn/docs/Zhong Ziwen_2022213356_MidTerm.pdf          # Mid-term report PDF submission
```

The `cursor_deep_learning_model_with_bonn_da.md` is the **most critical file** — it records the
full development history, every design decision, supervisor feedback, data leakage fix, and
experimental results. Read it thoroughly before suggesting any changes.

## 2. Project Overview

| Field | Value |
|-------|-------|
| Student | Zhong Ziwen |
| BUPT No. | 2022213356 |
| QM No. | 221168280 |
| Programme | Internet of Things Engineering |
| Title | AI Assisted Epilepsy Detection using EEG Signals |
| Dataset | University of Bonn EEG Dataset (5 subsets, 500 recordings) |
| Model | 1D-CNN + Bi-LSTM + Self-Attention (hybrid architecture) |
| Framework | PyTorch (migrated from TensorFlow due to Windows GPU issues) |

## 3. Critical Deadlines

| Item | Date |
|------|------|
| Draft final report | **13 Apr 2026** |
| Mock viva | 13–20 Apr 2026 |
| **Final report submission** | **27 Apr 2026** |
| Viva slides | 4 May 2026 |
| **Final viva** | **11–22 May 2026** |

## 4. Repository Structure

```
├── Bonn/                          # Model code & experiments
│   ├── config.py                  # Hyperparameters & paths
│   ├── data_loader.py             # Recording-level split + sliding window
│   ├── model.py                   # Hybrid, BiLSTM, CNN architectures
│   ├── train.py                   # Training loop (single/kfold)
│   ├── train_svm.py               # SVM + wavelet packet baseline
│   ├── predict.py                 # Inference script
│   ├── visualize.py               # EDA visualisations
│   ├── data/                      # Bonn dataset (Z/O/N/F/S subfolders)
│   ├── output/                    # K-Fold results, curves, confusion matrices
│   ├── eda/                       # EDA output plots
│   ├── docs/                      # ★ Project documents — READ FIRST
│   └── tests/                     # Unit tests
│
├── BBC6521_Final_Report_LaTeX_Template_25_26/  # LaTeX final report
│   ├── main.tex                   # Entry point (XeLaTeX only)
│   ├── cover.tex                  # Fill in student info
│   ├── contents/                  # Chapter .tex files
│   ├── appendices/                # Appendix .tex files
│   ├── reference.bib              # BibTeX references
│   ├── figures/                   # Report figures
│   └── requirements.sty           # DO NOT EDIT
```

## 5. Key Technical Decisions (Already Made)

These decisions have been validated. Do not reverse them without explicit user request:

1. **Recording-level K-Fold split** — prevents data leakage from overlapping sliding windows.
   Supervisor explicitly flagged segment-level splitting as unacceptable.
2. **Sliding window**: 1024 points (~5.9s), 50% overlap, applied AFTER train/val split per fold.
   Original Spec said 4s windows — explain the change in the report if asked.
3. **No additional bandpass filtering** — the Bonn dataset was already filtered at 0.53–40 Hz
   during acquisition (Andrzejak et al., 2001). The Spec mentioned "0.5–45 Hz filter" because
   it originally targeted CHB-MIT/TUH raw data; switching to Bonn made this unnecessary.
4. **StandardScaler fit on training set only** per fold. Uses global Z-Score (single mean/std
   across all data points in training set), which preserves inter-window amplitude differences.
5. **Three classification tasks**: Binary (Z+O vs S), Three-class (Z+O vs N+F vs S), Five-class (Z/O/N/F/S).
6. **Four model variants**: Hybrid (CNN+BiLSTM+Attn), Pure CNN, Pure BiLSTM+Attn, SVM+WPD.
7. **PyTorch with CUDA** — migrated from TensorFlow for Windows GPU support.

### Spec vs Implementation Deviations (document in report)

| Spec | Actual | Reason |
|------|--------|--------|
| CHB-MIT / TUH datasets | Bonn dataset | Smaller, cleaner, widely benchmarked |
| 0.5–45 Hz bandpass filter | No filter needed | Bonn already filtered at 0.53–40 Hz |
| 5-Fold CV | 10-Fold CV | More rigorous evaluation |
| 4-second windows | ~5.9s (1024 pts) | Better capture of seizure dynamics |
| TensorFlow/Keras | PyTorch | TF 2.11+ dropped Windows GPU support |

### Known Limitation (address in report)

Binary task class imbalance: Normal 200 recordings vs Seizure 100 (2:1 ratio after windowing).
No explicit handling (no `pos_weight`, no oversampling). Performance is unaffected (99.67%),
but should be acknowledged in the report discussion.

## 6. Latest Experiment Results (Recording-Level Split, No Leakage)

### Binary: Normal (Z+O) vs Seizure (S)
| Model | Accuracy | F1 |
|-------|----------|-----|
| SVM + Wavelet | 99.67% (±1.00) | 99.66% |
| Pure CNN | 98.86% (±1.91) | 98.84% |
| Pure BiLSTM | 99.48% (±1.29) | 99.47% |
| **Hybrid (Ours)** | **99.67% (±0.85)** | **99.66%** |

### Three-class: Normal vs Inter-ictal vs Seizure
| Model | Accuracy | F1 |
|-------|----------|-----|
| SVM | 94.20% (±2.44) | 94.19% |
| Pure CNN | 97.14% (±1.84) | 97.11% |
| Pure BiLSTM | 92.80% (±5.99) | 92.74% |
| **Hybrid (Ours)** | **97.69% (±2.17)** | **97.68%** |

### Five-class: Z vs O vs N vs F vs S
| Model | Accuracy | F1 |
|-------|----------|-----|
| SVM | 78.80% (±5.31) | 78.39% |
| Pure CNN | 78.86% (±3.77) | 75.95% |
| Pure BiLSTM | 73.09% (±8.37) | 71.92% |
| **Hybrid (Ours)** | **84.51% (±3.62)** | **84.28%** |

## 7. Supervisor Feedback (Critical)

The supervisor raised concerns about high binary accuracy (>99%). The response:
- Data leakage was found and fixed (segment-level → recording-level split)
- Post-fix binary 99.67% is consistent with literature (Thara 2019: 99%, Ullah 2018: 99.6%)
- Even SVM without sliding windows achieves 99.67% — task is inherently easy
- Five-class is the real differentiator for the hybrid model

## 8. Remaining Work (TODO)

- [ ] Attention weight visualisation (Chapter 4.3)
- [ ] Statistical significance testing — paired t-tests across folds (Chapter 4.4)
- [ ] Comparison with published literature table (Chapter 4.5)
- [ ] Complete final report LaTeX (all chapters)
- [ ] Prepare viva slides (15 min presentation + 10 min Q&A)
- [ ] Fill GenAI acknowledgement table (Cursor usage disclosure)

## 9. LaTeX Compilation

```bash
cd BBC6521_Final_Report_LaTeX_Template_25_26
latexmk -xelatex main.tex
```

Requires XeLaTeX (for xeCJK Chinese font support). BasicTeX + packages installed via Homebrew.

## 10. Rules When Working on This Project

1. **Always read `Bonn/docs/` first** to understand context before making changes.
2. **Never revert to segment-level data splitting** — supervisor explicitly forbids it.
3. **Use Vancouver referencing style** in the LaTeX report.
4. **Report length**: 30–50 pages (up to Chapter 5, excluding appendices).
5. **GenAI disclosure is mandatory** — document all AI tool usage honestly.
6. **Protected branches**: never push directly to main/master.
7. **Run tests** after code changes: `cd Bonn && python -m pytest tests/`.
8. When writing LaTeX, only edit files in `contents/`, `appendices/`, `reference.bib`, and `cover.tex`. Do NOT edit `requirements.sty` or `environments.sty`.
