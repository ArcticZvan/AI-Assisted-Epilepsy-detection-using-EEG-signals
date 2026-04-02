# AI-Assisted Epilepsy Detection using EEG Signals

Automated epilepsy seizure detection system based on the **Bonn University EEG dataset**, combining a hybrid deep learning architecture (1D-CNN + Bi-LSTM + Self-Attention) with traditional machine learning baselines (SVM + Wavelet Packet features).

Key design choice: **recording-level K-Fold cross-validation** to prevent data leakage — all sliding-window segments from the same EEG recording stay in the same fold.

## Quick Start

```bash
cd Bonn

# Install dependencies
pip install -r requirements.txt

# Run EDA visualizations
python visualize.py

# Train the hybrid model (binary classification, single split)
python train.py --task binary --model hybrid --mode single

# Full 10-Fold cross-validation
python train.py --task binary --model hybrid --mode kfold --folds 10

# SVM baseline
python train_svm.py --task binary --folds 10

# Inference on a single file
python predict.py --model_path saved_models/binary_hybrid_final.pt --input data/S/S001.txt --task binary
```

## Model Architecture

```
Input (1024, 1)
    ↓
Conv1D(64) → BN → ReLU → MaxPool → Dropout
    ↓
Conv1D(128) → BN → ReLU → MaxPool → Dropout
    ↓
Bi-LSTM(128, bidirectional) → Dropout
    ↓
Bi-LSTM(64, bidirectional) → Dropout
    ↓
Self-Attention (weighted temporal pooling)
    ↓
Dense(64) → ReLU → Dropout
    ↓
Output (Sigmoid / Softmax)
```

Three model variants are available for ablation study:
- **hybrid**: Full CNN + Bi-LSTM + Attention (recommended)
- **bilstm**: Bi-LSTM + Attention only (no CNN feature extraction)
- **cnn**: Pure 1D-CNN + Global Average Pooling (no temporal modeling)

## Experimental Results

| Task | Model | Accuracy | F1 (weighted) | AUC |
|------|-------|----------|---------------|-----|
| Binary (Normal vs Seizure) | Hybrid (CNN+BiLSTM+Attn) | — | — | — |
| Binary | Bi-LSTM + Attention | — | — | — |
| Binary | Pure 1D-CNN | — | — | — |
| Binary | SVM + Wavelet Packet | — | — | — |
| Three-class | Hybrid | — | — | — |
| Five-class | Hybrid | — | — | — |

> Fill in after running experiments. Expected: Binary > 98%, Three-class > 95%, Five-class > 90%.

## Dataset

The [Bonn University EEG dataset](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3) contains 5 subsets (100 recordings each, 4097 data points per recording at 173.61 Hz):

| Subset | Label | Description |
|--------|-------|-------------|
| Z (Set A) | Normal | Healthy volunteers, eyes open |
| O (Set B) | Normal | Healthy volunteers, eyes closed |
| N (Set C) | Inter-ictal | Epilepsy patients, seizure-free interval (contralateral) |
| F (Set D) | Inter-ictal | Epilepsy patients, seizure-free interval (epileptic zone) |
| S (Set E) | Ictal | Epilepsy patients, during seizure |

## References

1. Acharya et al. (2017) - Deep CNN for automated seizure detection
2. Hussein et al. (2018) - Robust LSTM for EEG seizure detection
3. Thara et al. (2019) - Stacked Bi-LSTM for seizure detection (99% on Bonn)
4. Ullah et al. (2018) - Pyramidal 1D-CNN for EEG classification

## License

[MIT](LICENSE)
