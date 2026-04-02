"""
Bonn EEG 数据集加载与预处理模块 (PyTorch 版)

修复数据泄露问题:
- K-Fold 在录音级别 (recording-level) 分割，而非段级别 (segment-level)
- 滑动窗口分割仅在分割后的训练/验证集上独立进行
- Z-Score 标准化仅 fit 训练集，transform 验证集
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import (
    SUBSET_FOLDERS, POINTS_PER_FILE,
    WINDOW_SIZE, WINDOW_STRIDE,
    BINARY_LABELS, THREE_CLASS_LABELS, FIVE_CLASS_LABELS,
    BATCH_SIZE,
)


LABEL_MAP_OPTIONS = {
    "binary": (BINARY_LABELS, ["Normal", "Seizure"]),
    "three": (THREE_CLASS_LABELS, ["Normal", "Inter-ictal", "Seizure"]),
    "five": (FIVE_CLASS_LABELS, ["Z (Normal EO)", "O (Normal EC)", "N (Inter-ictal)", "F (Inter-ictal)", "S (Seizure)"]),
}


def load_single_file(filepath: str) -> np.ndarray:
    """读取单个 EEG txt 文件，返回 1D numpy 数组。"""
    return np.loadtxt(filepath, dtype=np.float32)


def load_raw_dataset(subsets: list[str] | None = None) -> tuple[list[np.ndarray], list[str]]:
    """
    加载指定子集的原始 EEG 数据。

    Returns
    -------
    signals : list[np.ndarray], 每个元素 shape=(4097,)
    labels : list[str], 子集标签如 "Z", "S"
    """
    if subsets is None:
        subsets = list(SUBSET_FOLDERS.keys())

    signals = []
    labels = []

    for subset_name in subsets:
        folder = SUBSET_FOLDERS[subset_name]
        pattern = os.path.join(folder, f"{subset_name}*.txt")
        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            print(f"[WARNING] 子集 {subset_name} 在 {folder} 中未找到文件")
            continue

        for fpath in files:
            sig = load_single_file(fpath)
            if len(sig) != POINTS_PER_FILE:
                print(f"[WARNING] 文件 {fpath} 长度为 {len(sig)}，期望 {POINTS_PER_FILE}，已跳过")
                continue
            signals.append(sig)
            labels.append(subset_name)

    print(f"[INFO] 共加载 {len(signals)} 个原始录音")
    return signals, labels


def sliding_window_segment(signal: np.ndarray,
                           window_size: int = WINDOW_SIZE,
                           stride: int = WINDOW_STRIDE) -> np.ndarray:
    """对单个信号进行滑动窗口分割，返回 (num_windows, window_size)。"""
    num_windows = max(0, (len(signal) - window_size) // stride + 1)
    segments = np.zeros((num_windows, window_size), dtype=np.float32)
    for i in range(num_windows):
        start = i * stride
        segments[i] = signal[start:start + window_size]
    return segments


def load_recordings(task: str = "binary") -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    加载录音级别的数据 (不做滑动窗口分割)。

    Returns
    -------
    recordings : list[np.ndarray], 每个元素 shape=(4097,)
    rec_labels : np.ndarray, shape=(num_recordings,), 整数标签
    class_names : list[str]
    """
    if task not in LABEL_MAP_OPTIONS:
        raise ValueError(f"task 必须是 'binary', 'three', 'five' 之一，收到: {task}")

    label_map, class_names = LABEL_MAP_OPTIONS[task]
    subsets = list(label_map.keys())
    signals, str_labels = load_raw_dataset(subsets)

    recordings = signals
    rec_labels = np.array([label_map[lbl] for lbl in str_labels], dtype=np.int64)

    print(f"[INFO] 任务: {task} 分类")
    print(f"[INFO] 录音数: {len(recordings)}, 类别分布: {dict(zip(*np.unique(rec_labels, return_counts=True)))}")

    return recordings, rec_labels, class_names


def segment_and_normalize(train_recordings: list[np.ndarray],
                          train_labels: np.ndarray,
                          val_recordings: list[np.ndarray],
                          val_labels: np.ndarray,
                          window_size: int = WINDOW_SIZE,
                          stride: int = WINDOW_STRIDE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对已经分好组的训练/验证录音分别做滑动窗口分割和标准化。
    标准化 scaler 只在训练集上 fit，保证无数据泄露。

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    def _segment_group(recordings, labels):
        all_segs = []
        all_labels = []
        for sig, lbl in zip(recordings, labels):
            segs = sliding_window_segment(sig, window_size, stride)
            all_segs.append(segs)
            all_labels.extend([lbl] * len(segs))
        X = np.concatenate(all_segs, axis=0)
        y = np.array(all_labels, dtype=np.int64)
        return X, y

    X_train, y_train = _segment_group(train_recordings, train_labels)
    X_val, y_val = _segment_group(val_recordings, val_labels)

    scaler = StandardScaler()
    train_shape = X_train.shape
    val_shape = X_val.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(train_shape)
    X_val = scaler.transform(X_val.reshape(-1, 1)).reshape(val_shape)

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)

    return X_train, y_train, X_val, y_val


class EEGDataset(Dataset):
    """PyTorch Dataset 封装。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。"""
    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    recordings, labels, names = load_recordings(task="binary")
    print(f"\n二分类录音级别数据:")
    print(f"  录音数: {len(recordings)}")
    print(f"  每条录音长度: {recordings[0].shape}")
    print(f"  类别名: {names}")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(recordings, labels))
    train_recs = [recordings[i] for i in train_idx]
    train_lbls = labels[train_idx]
    val_recs = [recordings[i] for i in val_idx]
    val_lbls = labels[val_idx]

    X_tr, y_tr, X_va, y_va = segment_and_normalize(train_recs, train_lbls, val_recs, val_lbls)
    print(f"\n  训练段: X={X_tr.shape}, y={y_tr.shape}")
    print(f"  验证段: X={X_va.shape}, y={y_va.shape}")
    print(f"  无数据泄露: 训练录音 {len(train_idx)} 条, 验证录音 {len(val_idx)} 条, 完全不重叠")
