# Mock Viva Speech Script

**Total target: 10 minutes**

---

## Slide 1 — Title Page (~15s)

Good morning. My name is Zhong Ziwen. My project is titled "AI-Assisted Epilepsy Detection using EEG Signals". I'll walk you through the motivation, methods, and key findings today.

---

## Slide 2 — Outline (~10s)

Here's a quick overview. I'll cover the background, then the dataset and preprocessing, followed by the model design, experimental results, and finally the conclusion.

---

## Slide 3 — Problem and Motivation (~35s)

Epilepsy affects 50 million people worldwide. Diagnosis relies on neurologists manually reading EEG recordings — it's slow, subjective, and there aren't enough specialists to go around.

Automated EEG analysis can solve this. Methods have evolved from hand-crafted features with SVM, to CNNs, to LSTMs, and now to hybrid models. The idea behind a hybrid is straightforward: EEG signals have both local patterns — like spike shapes — and long-range temporal structure. CNNs are good at the local part, LSTMs are good at the temporal part, and attention helps the model focus on what matters most. So combining them makes sense. But existing hybrid models lack proper evaluation, which is what my project addresses.

---

## Slide 4 — Research Gaps and Objectives (~35s)

On the left, three gaps I found in the literature. First, data leakage — many studies split at the segment level, so overlapping windows from the same recording leak across train and test, inflating accuracy. Second, no ablation — hybrid models are proposed as a whole but nobody checks if each part actually helps. Third, most work is binary only — the interictal states get ignored.

On the right, my five objectives to address these gaps: recording-level cross-validation to prevent leakage; a lightweight CNN plus Bi-LSTM plus attention architecture; systematic ablation with three baselines; evaluation on binary, three-class, and five-class tasks; and attention visualisation for interpretability.

---

## Slide 5 — Dataset (~35s)

I used the University of Bonn EEG dataset — five subsets, Z through S, with 100 recordings each. Each recording is single-channel, sampled at about 174 hertz, lasting 23.6 seconds.

Z and O are from healthy subjects — eyes open and eyes closed. N and F are interictal recordings from epilepsy patients — contralateral and epileptogenic zones respectively. And S is the seizure set.

I set up three tasks of increasing difficulty: binary with 300 recordings, three-class and five-class with all 500. As you can see from the waveforms on the right, normal and seizure signals look very different, but the interictal ones are much more subtle.

---

## Slide 6 — Data Leakage (~35s)

This slide explains the data leakage problem, which is one of the key contributions of my project.

When you split at the segment level and use overlapping windows, segments from the same recording end up in both training and test. The model just memorises patterns instead of actually learning, so the accuracy looks great but means nothing.

My fix is simple but important: split at the recording level first, then do windowing. Each fold has 450 recordings for training and 50 for validation — completely separate. The scaler is also fit only on training data.

I actually caught this bug during development when my early results looked too good. After the fix, the numbers came down to realistic levels but stayed consistent with published benchmarks.

---

## Slide 7 — Preprocessing Pipeline (~20s)

This flowchart shows the full pipeline. Recordings go through recording-level 10-fold splitting, then sliding windows of 1024 points with 50 percent overlap, then Z-score normalisation fitted on training data only. No extra filtering is needed since the Bonn data was already filtered during acquisition.

---

## Slide 8 — Model Architecture (~40s)

Here's the proposed hybrid model. The input is a 1024-point EEG window. First, two 1D-CNN layers with 64 and 128 filters extract local features. Then two Bi-LSTM layers capture temporal dependencies in both directions. A self-attention layer assigns weights to each time step. And finally a dense layer with softmax gives the output. The whole model is only about 500K parameters.

Why this design? Because EEG signals have both local patterns — spikes and sharp waves that a CNN picks up well — and longer temporal dynamics — seizure discharges that build up over seconds, which is what the BiLSTM is for. And not every time step matters equally, so the attention layer learns to focus on the clinically important regions. As a bonus, we can visualise those attention weights, which gives us interpretability.

---

## Slide 9 — Ablation Study Design (~30s)

To prove each component matters, I tested four variants: the full Hybrid, a Pure CNN without LSTM or attention, a Pure Bi-LSTM plus attention without CNN, and an SVM baseline using wavelet features.

The key point is that all deep learning models use exactly the same data, same splits, same preprocessing — only the architecture changes. So any difference in performance is purely due to the model structure. Training used Adam with a learning rate of 1e-4 and early stopping.

---

## Slide 10 — Binary and Three-class Results (~40s)

Starting with binary classification — all models do well, above 98 percent. This makes sense because normal and seizure signals are very different. Not much to separate the models here.

But in the three-class task, adding the interictal category makes things harder. The Hybrid gets 97.69, CNN gets 97.14, but the BiLSTM alone drops to 92.80 with high variance — plus or minus 6. SVM is at 94.20. So as the task gets harder, the Hybrid's advantage starts to show, while single models become unstable.

---

## Slide 11 — Five-class Results (~40s)

The five-class task is where the real difference appears. The Hybrid achieves 84.51 percent — that's nearly 6 points above CNN and over 11 above BiLSTM. CNN catches local patterns but misses the bigger picture. BiLSTM captures sequences but lacks spatial precision. You need both working together, plus attention, to handle this level of complexity.

The confusion matrix shows the main errors come from N and F — both interictal, both recorded from epilepsy patients, very similar signals. Z and S are almost perfectly classified.

---

## Slide 12 — Attention Visualisation (~30s)

This heatmap shows what the attention layer focuses on. For seizure signals, it puts high weights on sharp waves and high-frequency bursts — exactly what neurologists look for. For normal signals, the weights are spread evenly — nothing abnormal to highlight. For interictal signals, there are moderate peaks on subtle spike-and-wave patterns.

This matters because it makes the model interpretable. Doctors can see why the model made its decision, which is essential for building clinical trust.

---

## Slide 13 — Statistical Testing and Literature Comparison (~30s)

I ran paired t-tests to make sure these results aren't just luck. The Hybrid beats CNN with p equals 0.0008, BiLSTM with p equals 0.003, and SVM with p equals 0.023 — all statistically significant.

Compared to published work, our 84.5 percent on five-class is 17 points higher than Huang et al.'s 67.2 percent on the same benchmark. And unlike some studies that may have leakage issues, our recording-level split guarantees a fair comparison.

---

## Slide 14 — Limitations and Further Work (~25s)

I should acknowledge some limitations. The Bonn dataset is small — 500 recordings from just 10 subjects. We only use single-channel EEG. The binary task has a 2-to-1 imbalance we didn't explicitly handle. And it's an offline system, not real-time.

Going forward, I'd like to validate on larger multi-channel datasets like CHB-MIT and TUH, try hyperparameter optimisation, explore transfer learning, and work towards real-time deployment on wearable devices.

---

## Slide 15 — Summary (~15s)

To wrap up — four contributions: a leakage-free evaluation framework, a hybrid model that reaches 84.5 percent on five-class and statistically beats all baselines, a proper ablation proving each component matters, and interpretable attention that aligns with clinical knowledge.

Thank you. I'm happy to take questions.

---

**Approximate timing breakdown:**

| Section | Slides | Time |
|---------|--------|------|
| Title + Outline | 1–2 | ~25s |
| Background & Motivation | 3–4 | ~1m 10s |
| Dataset & Preprocessing | 5–7 | ~1m 30s |
| Proposed Method | 8–9 | ~1m 20s |
| Results & Discussion | 10–13 | ~2m 20s |
| Conclusion | 14–15 | ~40s |
| **Total** | **1–15** | **~8m 25s** |

> Buffer of ~1m 35s for natural pauses, transitions, and audience eye contact.
