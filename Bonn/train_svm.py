"""
Bonn EEG 癫痫检测 - SVM + Wavelet Packet 基线模型

传统机器学习方法:
1. 对 EEG 信号进行小波包分解 (Wavelet Packet Decomposition)
2. 提取各子带的统计特征 (能量、均值、标准差、熵等)
3. 使用 SVM (RBF 核) 进行分类
4. 10-Fold 交叉验证
"""
import os
import json
import argparse
import numpy as np
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from config import (
    K_FOLDS, RANDOM_SEED, OUTPUT_DIR,
    BINARY_LABELS, THREE_CLASS_LABELS, FIVE_CLASS_LABELS,
)
from data_loader import load_raw_dataset


def wavelet_packet_features(signal: np.ndarray,
                            wavelet: str = "db4",
                            level: int = 4) -> np.ndarray:
    """
    对单个信号提取小波包特征。

    对每个子带节点计算:
    - 能量 (energy)
    - 均值 (mean)
    - 标准差 (std)
    - 香农熵 (Shannon entropy)

    Parameters
    ----------
    signal : np.ndarray, shape=(N,)
    wavelet : str, 小波基函数
    level : int, 分解层数

    Returns
    -------
    features : np.ndarray, shape=(num_nodes * 4,)
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, order="freq")]

    features = []
    for node_path in nodes:
        coeffs = wp[node_path].data.astype(np.float64)

        energy = np.sum(coeffs ** 2)
        mean_val = np.mean(coeffs)
        std_val = np.std(coeffs)

        coeffs_sq = coeffs ** 2
        total = np.sum(coeffs_sq)
        if total > 0:
            prob = coeffs_sq / total
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
        else:
            entropy = 0.0

        features.extend([energy, mean_val, std_val, entropy])

    return np.array(features, dtype=np.float64)


def extract_features_dataset(task: str = "binary") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    对整个数据集提取小波包特征 (基于完整信号，不做滑动窗口)。
    """
    label_map_options = {
        "binary": (BINARY_LABELS, ["Normal", "Seizure"]),
        "three": (THREE_CLASS_LABELS, ["Normal", "Inter-ictal", "Seizure"]),
        "five": (FIVE_CLASS_LABELS, ["Z (Normal EO)", "O (Normal EC)", "N (Inter-ictal)", "F (Inter-ictal)", "S (Seizure)"]),
    }

    label_map, class_names = label_map_options[task]
    subsets = list(label_map.keys())
    signals, str_labels = load_raw_dataset(subsets)

    print("[INFO] 正在提取小波包特征 (可能需要几秒)...")
    X_list = []
    y_list = []
    for sig, lbl in zip(signals, str_labels):
        feat = wavelet_packet_features(sig)
        X_list.append(feat)
        y_list.append(label_map[lbl])

    X = np.array(X_list)
    y = np.array(y_list, dtype=np.int64)

    print(f"[INFO] 任务: {task} 分类")
    print(f"[INFO] 特征矩阵: {X.shape} (样本数 x 特征数)")
    print(f"[INFO] 类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, class_names


def plot_confusion_matrix(y_true, y_pred, class_names, fold, output_dir: str):
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SVM Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold{fold}.png"), dpi=150)
    plt.close()


def train_svm_kfold(task: str = "binary", n_folds: int = K_FOLDS):
    """SVM + Wavelet Packet 特征的 K-Fold 交叉验证。"""
    np.random.seed(RANDOM_SEED)

    task_output_dir = os.path.join(OUTPUT_DIR, f"{task}_svm")
    os.makedirs(task_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SVM Baseline: {task} 分类, {n_folds}-Fold CV")
    print(f"{'='*60}\n")

    X, y, class_names = extract_features_dataset(task=task)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_accuracies = []
    fold_f1_scores = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n  Fold {fold}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_SEED)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"    Accuracy={acc:.4f}, F1={f1:.4f}")
        print(classification_report(y_val, y_pred, target_names=class_names))

        plot_confusion_matrix(y_val, y_pred, class_names, fold, task_output_dir)

    print(f"\n{'='*60}")
    print(f"  SVM {n_folds}-Fold 交叉验证汇总")
    print(f"{'='*60}")
    print(f"  平均 Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"  平均 F1 Score: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
    print(f"  各折 Accuracy: {[f'{a:.4f}' for a in fold_accuracies]}")

    plot_confusion_matrix(all_y_true, all_y_pred, class_names, fold="all", output_dir=task_output_dir)

    results = {
        "task": task,
        "model_type": "SVM + Wavelet Packet",
        "n_folds": n_folds,
        "fold_accuracies": fold_accuracies,
        "fold_f1_scores": fold_f1_scores,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_f1": float(np.mean(fold_f1_scores)),
        "std_f1": float(np.std(fold_f1_scores)),
    }

    results_path = os.path.join(task_output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {task_output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="SVM + Wavelet Packet 基线训练")
    parser.add_argument(
        "--task", type=str, default="binary",
        choices=["binary", "three", "five"],
        help="分类任务: binary(二分类), three(三分类), five(五分类)",
    )
    parser.add_argument(
        "--folds", type=int, default=K_FOLDS,
        help=f"K折交叉验证折数 (默认: {K_FOLDS})",
    )
    args = parser.parse_args()
    train_svm_kfold(task=args.task, n_folds=args.folds)


if __name__ == "__main__":
    main()
