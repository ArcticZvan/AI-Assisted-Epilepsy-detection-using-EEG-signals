"""
Attention weight visualisation for the hybrid CNN-BiLSTM-Attention model.

Loads a saved model checkpoint, runs CPU inference on representative EEG
recordings, and plots attention weight heatmaps to show which temporal
regions the model focuses on for each class.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, WINDOW_SIZE, WINDOW_STRIDE

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

FIVE_CLASS_LABELS = {"Z": 0, "O": 1, "N": 2, "F": 3, "S": 4}
FIVE_CLASS_NAMES = ["Z (Normal EO)", "O (Normal EC)", "N (Inter-ictal)",
                    "F (Inter-ictal)", "S (Seizure)"]


def _load_single_file(filepath: str) -> np.ndarray:
    return np.loadtxt(filepath, dtype=np.float32)


def _sliding_window(signal: np.ndarray, window: int = WINDOW_SIZE,
                    stride: int = WINDOW_STRIDE) -> np.ndarray:
    n = max(0, (len(signal) - window) // stride + 1)
    out = np.zeros((n, window), dtype=np.float32)
    for i in range(n):
        out[i] = signal[i * stride: i * stride + window]
    return out


def _build_hybrid_model(num_classes: int = 5):
    """Lazy import to handle Python 3.9 compatibility."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "model", os.path.join(BASE_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__builtins__ = __builtins__
    import types
    old_annotations = getattr(types, '__future__', None)
    exec(compile(
        open(os.path.join(BASE_DIR, "model.py")).read(),
        os.path.join(BASE_DIR, "model.py"),
        "exec",
        flags=__import__('__future__').annotations.compiler_flag,
    ), mod.__dict__)
    return mod.build_model("hybrid", num_classes=num_classes)

SAMPLE_RECORDINGS = {
    "Z": os.path.join(DATA_DIR, "Z", "Z001.txt"),
    "S": os.path.join(DATA_DIR, "S", "S001.txt"),
    "N": os.path.join(DATA_DIR, "N", "N001.txt"),
    "F": os.path.join(DATA_DIR, "F", "F001.txt"),
}

MODEL_PATH = os.path.join(OUTPUT_DIR, "five_hybrid", "best_model_fold1.pt")


def load_and_preprocess(filepath: str) -> np.ndarray:
    signal = _load_single_file(filepath)
    segments = _sliding_window(signal, WINDOW_SIZE, WINDOW_STRIDE)
    mean, std = segments.mean(), segments.std()
    if std < 1e-8:
        std = 1.0
    segments = (segments - mean) / std
    return segments[..., np.newaxis].astype(np.float32)


def get_attention_weights(model, segments: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(segments).float()
        _, attn_weights = model(x)
    return attn_weights.numpy()


def reconstruct_full_attention(attn_weights: np.ndarray,
                               signal_length: int = 4097) -> np.ndarray:
    """Reconstruct a full-length attention signal by averaging overlapping windows."""
    n_windows, seq_len = attn_weights.shape
    full_attn = np.zeros(signal_length, dtype=np.float64)
    counts = np.zeros(signal_length, dtype=np.float64)

    cnn_downsample = WINDOW_SIZE // seq_len
    for i in range(n_windows):
        start = i * WINDOW_STRIDE
        for j in range(seq_len):
            t_start = start + j * cnn_downsample
            t_end = min(t_start + cnn_downsample, signal_length)
            full_attn[t_start:t_end] += attn_weights[i, j]
            counts[t_start:t_end] += 1

    mask = counts > 0
    full_attn[mask] /= counts[mask]
    return full_attn


def plot_attention_heatmap(save_path: str) -> None:
    model = _build_hybrid_model(num_classes=5)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded model from {MODEL_PATH}")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(len(SAMPLE_RECORDINGS), 1, hspace=0.35)

    sample_rate = 173.61

    for idx, (subset, fpath) in enumerate(SAMPLE_RECORDINGS.items()):
        signal = _load_single_file(fpath)
        segments = load_and_preprocess(fpath)
        attn = get_attention_weights(model, segments)
        full_attn = reconstruct_full_attention(attn, len(signal))

        time_axis = np.arange(len(signal)) / sample_rate
        label = FIVE_CLASS_NAMES[FIVE_CLASS_LABELS[subset]]

        ax = fig.add_subplot(gs[idx])
        ax.plot(time_axis, signal / np.max(np.abs(signal)), color="#888888",
                alpha=0.5, linewidth=0.5, label="EEG (normalised)")

        attn_norm = (full_attn - full_attn.min())
        amax = attn_norm.max()
        if amax > 0:
            attn_norm /= amax
        ax.fill_between(time_axis, 0, attn_norm, alpha=0.6, color="#E74C3C",
                         label="Attention weight")

        ax.set_title(f"{label}  ({os.path.basename(fpath)})", fontsize=12, fontweight="bold")
        ax.set_xlim(0, time_axis[-1])
        ax.set_ylim(-0.1, 1.15)
        ax.set_ylabel("Normalised")
        if idx == len(SAMPLE_RECORDINGS) - 1:
            ax.set_xlabel("Time (seconds)")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Self-Attention Weights over EEG Signals (Five-class Hybrid Model)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved attention heatmap to {save_path}")


if __name__ == "__main__":
    out_path = os.path.join(OUTPUT_DIR, "attention_heatmap.png")
    plot_attention_heatmap(out_path)
