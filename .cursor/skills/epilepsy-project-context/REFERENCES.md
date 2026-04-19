# Verified Reference Details

All 14 papers in `reference.bib` have been verified via WebSearch. This file records
the key facts from each paper so agents can cite them accurately without re-searching.

## andrzejak2001 — Bonn Dataset Original Paper
- **Journal**: Physical Review E, 64(6), 061907, 2001
- **Dataset**: 5 sets (A–E), 100 single-channel EEG segments each, 23.6s, 173.61 Hz, 4097 points
- **Bandpass filter**: 0.53–40 Hz (12 dB/oct) applied during acquisition
- **Key finding**: Strongest nonlinear deterministic dynamics found in seizure activity (Set E)
- **Surface EEG**: Sets A (eyes open), B (eyes closed) from 5 healthy volunteers
- **Intracranial EEG**: Sets C (contralateral), D (epileptogenic zone), E (seizure) from 5 patients

## acharya2018 — Deep CNN for Seizure Detection
- **Journal**: Computers in Biology and Medicine, 100, 270–278, 2018
- **Architecture**: 13-layer deep CNN
- **Dataset**: Bonn, three-class (normal/preictal/seizure)
- **Accuracy**: 88.67% (three-class, 10-fold CV), Sensitivity 95%, Specificity 90%
- **Note**: Among the first to apply CNN directly to raw EEG; accuracy modest for three-class

## ullah2018 — Pyramidal 1D-CNN
- **Journal**: Expert Systems with Applications, 107, 61–71, 2018
- **Architecture**: Ensemble of Pyramidal 1D-CNNs (P-1D-CNN), 60% fewer parameters than standard CNN
- **Dataset**: Bonn
- **Accuracy**: 99.1% (±0.9%) on binary, improved ternary over prior SOTA of 97±1%
- **Data augmentation**: Two augmentation schemes proposed for small datasets

## thara2019 — Stacked Bi-LSTM [PDF in Bonn/docs/Stacked Bi-LSTM Bonn.pdf]
- **Journal**: Pattern Recognition Letters, 128, 529–535, 2019
- **Authors**: D.K. Thara, B.G. PremaSudha, Fan Xiong
- **Architecture**: Stacked bidirectional LSTM (2 layers)
- **Dataset**: Bonn, split into 23 segments of 178 points each per recording
- **Detection accuracy**: 99.08% (precision 98%, recall 99.5%, AUC 0.984)
- **Also did prediction**: Preictal state classification, sensitivity 89.21%, FPR 0.06
- **Data leakage issue**: All 11,500 segments shuffled before 80/20 split — segments
  from same recording can appear in both train and test sets. Paper does not discuss this.
- **Practical vision**: Proposed wearable wrist-watch device for seizure prediction alarm

## hussein2019 — Optimised LSTM for Robust Detection
- **Journal**: Clinical Neurophysiology, 130(1), 25–37, 2019
- **Architecture**: Optimised LSTM with regularisation
- **Focus**: Robustness against EEG artefacts (muscle activity, eye movements, noise)
- **Key contribution**: Segmentation + average pooling strategy for non-stationary signals
- **Dataset**: Bonn and others

## shoeibi2021 — DL for Seizure Detection Review [SUPERVISOR-RECOMMENDED]
- **Journal**: IJERPH, 18(11), 5780, 2021 (Open Access, CC BY)
- **Authors**: Afshin Shoeibi + 13 co-authors from Iran, Australia, USA, Egypt, Singapore, Taiwan
- **Type**: Comprehensive review of DL methods for EEG/MRI-based seizure detection
- **Covers**: CNN, RNN, LSTM, GAN, AE, hybrid architectures, rehabilitation systems, cloud/hardware
- **Key finding**: Hybrid architectures (CNN+RNN) consistently outperform single-architecture models
- **Gap noted**: Most studies lack systematic ablation experiments
- **Source**: PDF provided by supervisor, read from `Bonn/docs/Epileptic Seizures Detection Using Deep Learning Techniques A Review.pdf`. VERIFIED.

## huang2025 — STFFDA Dual Attention Model [PDF in Bonn/docs/EEG detection and recognition model...pdf]
- **Journal**: Scientific Reports, 15, 9404, 2025
- **Authors**: Zhentao Huang, Yuyao Yang, Yahong Ma, Qi Dong, Jianyun Su, Hangyu Shi, Shanwen Zhang, Liangliang Hu
- **Architecture**: Two parallel modules: CNN (spatial features) + 1D-SE attention, Bi-LSTM
  (temporal features) + dot-product attention. Raw EEG input, no preprocessing needed.
- **Datasets**: Bonn + CHB-MIT
- **Bonn results**: 77.65% single-validation, 67.24% 10-Fold CV (five-class)
- **CHB-MIT results**: 95.18% single-validation, 92.42% 10-Fold CV
- **Key contribution**: End-to-end model eliminating need for manual feature extraction
- **Limitation**: Treats each 4097-point recording as set of 178-dim samples, discarding
  most temporal context. Different preprocessing from our approach makes direct comparison indicative.
- **Our advantage**: 84.51% vs 67.24% on same Bonn five-class 10-Fold benchmark (+17 points)

## chen2023 — RF + CNN Feature Fusion [PDF in Bonn/docs/An_automated_detection_of_epileptic_seizures_EEG_u.pdf]
- **Journal**: BMC Medical Informatics and Decision Making, 23, 96, 2023 (Open Access, CC BY 4.0)
- **Authors**: Wenna Chen, Yixing Wang, Yuhao Ren, Hongwei Jiang, Ganqin Du, Jincan Zhang, Jinghua Li
- **Method**: DWT decomposition → extract ApEn + FuzzyEn + SampEn + STD per subband →
  Random Forest for feature selection (reduces redundancy) → CNN classifier
- **Datasets**: Bonn + New Delhi
- **Bonn interictal vs ictal**: 99.9% accuracy, 100% sensitivity, 99.81% precision, 99.8% specificity
- **New Delhi interictal vs ictal**: 100% on all metrics
- **ABCD-E (non-seizure vs seizure)**: 98.47%
- **Limitation**: Only binary classification tasks, no five-class, no cross-validation (3:1 split)
- **Relevance**: Their wavelet + entropy feature approach is conceptually similar to our SVM
  baseline's wavelet-packet features, but they add RF selection + CNN instead of SVM

## goldberger2000 — PhysioBank/PhysioNet
- **Journal**: Circulation, 101(23), e215–e220, 2000
- **Type**: Resource paper describing PhysioBank (data), PhysioToolkit (software), PhysioNet (portal)
- **Relevance**: Platform through which the Bonn dataset is publicly available

## shoeb2010 — ML for Seizure Detection (CHB-MIT)
- **Venue**: ICML 2010, Haifa, Israel
- **Method**: Patient-specific SVM classifiers on scalp EEG
- **Dataset**: CHB-MIT (916 hours from 24 patients)
- **Results**: 96% detection of 173 seizures, 3s median delay, 2 false alarms/24h
- **Note**: Established CHB-MIT as standard benchmark; patient-specific limits scalability

## obeid2016 — TUH EEG Corpus
- **Journal**: Frontiers in Neuroscience, 10, 196, 2016
- **Dataset**: 20,000+ EEGs from Temple University Hospital (2002–2013+)
- **Features**: Multi-channel clinical EEG with physician reports, ICD-9 codes
- **Relevance**: Largest public clinical EEG corpus; target for future work validation

## zhang2026 — CodimNet (Alzheimer's) [SUPERVISOR-RECOMMENDED — DO NOT REMOVE]
- **Journal**: Expert Systems with Applications, 297, 129353, 2026
- **Authors**: Zhengliang Zhang, Yachen Wei, Xin Rao, Liyang Yu, Wen Sun, Ruixue Li, Xiaoshuai Zhang, Xiaodong Chen, Xingru Huang
- **Affiliations**: Hangzhou Dianzi University, Ocean University of China, University of Liverpool
- **Method**: CodimNet — neuroanatomically stratified cortical partitioning (region-specific
  feature extraction from 19-channel EEG) + Neurodynamic PINN (ND-PINN) for spectral
  regularization enforcing biologically informed constraints on frequency-domain PSD
- **Application**: Alzheimer's disease early detection
- **Results**: 76.27% on CAUEEG (largest public AD EEG dataset), 80.68% on Thessaloniki, 72.57% on Tehran
- **Outperforms**: EEGNet, ShallowConvNet and other mainstream AD detection benchmarks
- **Code**: https://github.com/IMOP-lab/CodimNet
- **Relevance to project**: Demonstrates that incorporating brain-structure domain knowledge
  into network architecture improves EEG classification. Influenced our CNN+LSTM separation.
- **Source**: PDF provided by supervisor, also in Early-term report [2]. Full text read from
  `Bonn/docs/CodimNet.pdf`. VERIFIED.

## roy2019 — ChronoNet
- **Venue**: AIME 2019 (Springer)
- **Architecture**: 1D convolutions (exponentially varying filter lengths) + densely connected GRU layers
- **Dataset**: TUH Abnormal EEG Corpus
- **Result**: Outperformed prior SOTA by 7.79 percentage points
- **Note**: Domain-independent (also works on speech commands); processes raw time-series

## golmohammadi2019 — GRU vs LSTM for Seizure Detection
- **Venue**: IEEE Signal Processing in Medicine and Biology Symposium, 2019
- **Method**: Hybrid CNN-RNN comparing LSTM vs GRU units
- **Dataset**: TUH EEG Seizure Corpus (TUSZ)
- **Key finding**: Conv-LSTM significantly outperforms Conv-GRU
- **Result**: 30% sensitivity at 6 false alarms/24h (clinically relevant metric)
- **Validated on**: Duke University Seizure Corpus (cross-institution generalisation)
