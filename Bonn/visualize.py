"""
Bonn EEG 数据集 - 可视化与探索性数据分析 (EDA)

功能:
1. 各子集信号波形对比
2. 频域分析 (FFT)
3. 数据集分布统计
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SUBSET_FOLDERS, SAMPLE_RATE, BASE_DIR
from data_loader import load_raw_dataset

EDA_DIR = os.path.join(BASE_DIR, "eda")


def plot_signal_comparison(output_dir: str = EDA_DIR):
    """绘制五个子集的代表性信号波形对比图。"""
    os.makedirs(output_dir, exist_ok=True)

    subset_names = ["Z", "O", "N", "F", "S"]
    subset_titles = [
        "Set A (Z) - Normal, Eyes Open",
        "Set B (O) - Normal, Eyes Closed",
        "Set C (N) - Inter-ictal (Contralateral)",
        "Set D (F) - Inter-ictal (Epileptic Zone)",
        "Set E (S) - Ictal (Seizure)",
    ]

    signals, labels = load_raw_dataset(subset_names)

    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (name, title) in enumerate(zip(subset_names, subset_titles)):
        idx = labels.index(name)
        sig = signals[idx]
        time = np.arange(len(sig)) / SAMPLE_RATE

        axes[i].plot(time, sig, color=colors[i], linewidth=0.5, alpha=0.8)
        axes[i].set_ylabel("Amplitude (μV)")
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)

        amp_range = np.max(np.abs(sig))
        axes[i].text(
            0.98, 0.95, f"Max |Amp|: {amp_range:.0f}",
            transform=axes[i].transAxes,
            ha="right", va="top",
            fontsize=10, color=colors[i],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.suptitle("Bonn EEG Dataset - Signal Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 信号对比图已保存")


def plot_fft_comparison(output_dir: str = EDA_DIR):
    """绘制各子集的频域分析 (FFT) 对比图。"""
    os.makedirs(output_dir, exist_ok=True)

    subset_names = ["Z", "O", "N", "F", "S"]
    signals, labels = load_raw_dataset(subset_names)

    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    subset_titles = ["Set A (Z)", "Set B (O)", "Set C (N)", "Set D (F)", "Set E (S)"]

    for i, name in enumerate(subset_names):
        idx = labels.index(name)
        sig = signals[idx]

        fft_vals = np.fft.rfft(sig)
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / SAMPLE_RATE)
        magnitude = np.abs(fft_vals)

        mask = fft_freqs <= 60
        axes[i].plot(fft_freqs[mask], magnitude[mask], color=colors[i], linewidth=0.8)
        axes[i].set_ylabel("Magnitude")
        axes[i].set_title(f"{subset_titles[i]} - Frequency Spectrum", fontsize=12)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=12)
    plt.suptitle("Bonn EEG Dataset - FFT Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fft_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] FFT 对比图已保存")


def plot_amplitude_distribution(output_dir: str = EDA_DIR):
    """绘制各子集振幅分布直方图。"""
    os.makedirs(output_dir, exist_ok=True)

    subset_names = ["Z", "O", "N", "F", "S"]
    signals, labels = load_raw_dataset(subset_names)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, name in enumerate(subset_names):
        subset_signals = [signals[j] for j in range(len(labels)) if labels[j] == name]
        all_values = np.concatenate(subset_signals)

        axes[i].hist(all_values, bins=100, color=colors[i], alpha=0.7, density=True)
        axes[i].set_title(f"Set {name}", fontsize=12)
        axes[i].set_xlabel("Amplitude (μV)")
        axes[i].grid(True, alpha=0.3)

        axes[i].text(
            0.95, 0.95,
            f"μ={np.mean(all_values):.1f}\nσ={np.std(all_values):.1f}",
            transform=axes[i].transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes[0].set_ylabel("Density")
    plt.suptitle("Amplitude Distribution per Subset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amplitude_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 振幅分布图已保存")


if __name__ == "__main__":
    print("开始 EDA 可视化分析...")
    plot_signal_comparison()
    plot_fft_comparison()
    plot_amplitude_distribution()
    print(f"\n所有可视化结果已保存到 {EDA_DIR}")
