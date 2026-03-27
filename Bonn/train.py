"""
Bonn EEG 癫痫检测 - PyTorch 训练与评估脚本

修复数据泄露: K-Fold 在录音级别 (recording-level) 分割
- 同一条录音的所有滑动窗口段只出现在训练集或验证集中，绝不交叉
- 标准化只在训练集上 fit
"""
import os
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    BATCH_SIZE, EPOCHS, K_FOLDS, RANDOM_SEED,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    WINDOW_SIZE, LEARNING_RATE, OUTPUT_DIR, MODEL_DIR,
)
from data_loader import load_recordings, segment_and_normalize, get_dataloaders
from model import build_model, count_parameters


def set_seed(seed: int = RANDOM_SEED):
    """设置全局随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动检测并返回最佳设备。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] 使用: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[CPU] 未检测到 GPU，使用 CPU 训练")
    return device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 loss 和 accuracy。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)

        if model.num_classes == 2:
            loss = criterion(logits, y_batch.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
        else:
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型，返回 loss, accuracy, 预测值和真实值。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits, _ = model(X_batch)

        if model.num_classes == 2:
            loss = criterion(logits, y_batch.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
        else:
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)

        total_loss += loss.item() * X_batch.size(0)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          fold, output_dir: str):
    """绘制训练/验证 loss 和 accuracy 曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(val_accs, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Fold {fold} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curve_fold{fold}.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, fold, output_dir: str):
    """绘制混淆矩阵热力图。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold{fold}.png"), dpi=150)
    plt.close()


def plot_combined_training_curves(all_histories: list[dict], output_dir: str):
    """将所有 fold 的训练曲线整合到一张图中: 各 fold 浅色细线 + 均值粗线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.tab10
    max_epochs = max(len(h["train_losses"]) for h in all_histories)

    for idx, h in enumerate(all_histories):
        epochs = range(1, len(h["train_losses"]) + 1)
        color = cmap(idx % 10)
        axes[0].plot(epochs, h["val_losses"], color=color, alpha=0.25, linewidth=0.8)
        axes[1].plot(epochs, h["val_accs"], color=color, alpha=0.25, linewidth=0.8)

    padded_val_losses = np.full((len(all_histories), max_epochs), np.nan)
    padded_val_accs = np.full((len(all_histories), max_epochs), np.nan)
    padded_train_losses = np.full((len(all_histories), max_epochs), np.nan)
    padded_train_accs = np.full((len(all_histories), max_epochs), np.nan)

    for idx, h in enumerate(all_histories):
        n = len(h["train_losses"])
        padded_val_losses[idx, :n] = h["val_losses"]
        padded_val_accs[idx, :n] = h["val_accs"]
        padded_train_losses[idx, :n] = h["train_losses"]
        padded_train_accs[idx, :n] = h["train_accs"]

    epochs_range = np.arange(1, max_epochs + 1)
    mean_train_loss = np.nanmean(padded_train_losses, axis=0)
    mean_val_loss = np.nanmean(padded_val_losses, axis=0)
    mean_train_acc = np.nanmean(padded_train_accs, axis=0)
    mean_val_acc = np.nanmean(padded_val_accs, axis=0)

    axes[0].plot(epochs_range, mean_train_loss, "b-", linewidth=2, label="Train Loss (mean)")
    axes[0].plot(epochs_range, mean_val_loss, "r-", linewidth=2, label="Val Loss (mean)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss (All Folds)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, mean_train_acc, "b-", linewidth=2, label="Train Acc (mean)")
    axes[1].plot(epochs_range, mean_val_acc, "r-", linewidth=2, label="Val Acc (mean)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy (All Folds)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves_combined.png"), dpi=200)
    plt.close()
    print(f"  合并训练曲线已保存: {os.path.join(output_dir, 'training_curves_combined.png')}")


def run_training(model, train_loader, val_loader, num_classes,
                 device, fold, output_dir):
    """
    完整训练循环: 训练 + 验证 + 早停 + 学习率衰减。
    Returns: (best_val_acc, best_val_f1, y_true, y_pred)
    """
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=False,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    model_path = os.path.join(output_dir, f"best_model_fold{fold}.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    model.load_state_dict(best_model_state)
    _, best_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)
    best_f1 = f1_score(y_true, y_pred, average="weighted")

    plot_training_history(train_losses, val_losses, train_accs, val_accs, fold, output_dir)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }
    hist_path = os.path.join(output_dir, f"history_fold{fold}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    return best_acc, best_f1, y_true, y_pred, history


def _split_recordings(recordings, rec_labels, train_idx, val_idx):
    """按索引分割录音。"""
    train_recs = [recordings[i] for i in train_idx]
    train_lbls = rec_labels[train_idx]
    val_recs = [recordings[i] for i in val_idx]
    val_lbls = rec_labels[val_idx]
    return train_recs, train_lbls, val_recs, val_lbls


def train_single_split(task: str = "binary", model_type: str = "hybrid",
                        test_ratio: float = 0.2):
    """
    单次 train/test 分割 (Recording-level)。
    """
    set_seed()
    device = get_device()

    task_output_dir = os.path.join(OUTPUT_DIR, f"{task}_{model_type}_single")
    os.makedirs(task_output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"快速训练: {task} 分类, 模型: {model_type}, GPU: {torch.cuda.is_available()}")
    print(f"[Recording-level split — 无数据泄露]")
    print(f"{'='*60}\n")

    recordings, rec_labels, class_names = load_recordings(task=task)
    num_classes = len(class_names)

    rec_indices = np.arange(len(recordings))
    train_idx, val_idx = train_test_split(
        rec_indices, test_size=test_ratio, random_state=RANDOM_SEED, stratify=rec_labels,
    )
    print(f"录音分割: 训练 {len(train_idx)} 条, 验证 {len(val_idx)} 条 (无重叠)")

    train_recs, train_lbls, val_recs, val_lbls = _split_recordings(
        recordings, rec_labels, train_idx, val_idx,
    )
    X_train, y_train, X_val, y_val = segment_and_normalize(
        train_recs, train_lbls, val_recs, val_lbls,
    )
    print(f"分割后段数: 训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}")

    train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val)

    model = build_model(model_type, num_classes=num_classes).to(device)
    print(f"模型参数量: {count_parameters(model):,}\n")

    acc, f1, y_true, y_pred, _ = run_training(
        model, train_loader, val_loader, num_classes, device, fold=0, output_dir=task_output_dir,
    )

    print(f"\n单次分割结果: Accuracy={acc:.4f}, F1={f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names, fold=0, output_dir=task_output_dir)

    final_path = os.path.join(MODEL_DIR, f"{task}_{model_type}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"模型已保存到: {final_path}")

    return model


def train_kfold(task: str = "binary", model_type: str = "hybrid", n_folds: int = K_FOLDS):
    """
    K-Fold 交叉验证 (Recording-level split — 无数据泄露)。
    """
    set_seed()
    device = get_device()

    task_output_dir = os.path.join(OUTPUT_DIR, f"{task}_{model_type}")
    os.makedirs(task_output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"开始训练: {task} 分类, 模型: {model_type}, {n_folds}-Fold CV")
    print(f"[Recording-level split — 无数据泄露]")
    print(f"GPU: {torch.cuda.is_available()}")
    print(f"{'='*60}\n")

    recordings, rec_labels, class_names = load_recordings(task=task)
    num_classes = len(class_names)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_accuracies = []
    fold_f1_scores = []
    all_y_true = []
    all_y_pred = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(recordings, rec_labels), start=1):
        print(f"\n{'─'*40}")
        print(f"  Fold {fold}/{n_folds}")
        print(f"{'─'*40}")

        train_recs, train_lbls, val_recs, val_lbls = _split_recordings(
            recordings, rec_labels, train_idx, val_idx,
        )
        print(f"  录音: 训练 {len(train_idx)} 条, 验证 {len(val_idx)} 条")

        X_train, y_train, X_val, y_val = segment_and_normalize(
            train_recs, train_lbls, val_recs, val_lbls,
        )
        print(f"  段数: 训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}")

        train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val)

        model = build_model(model_type, num_classes=num_classes).to(device)
        if fold == 1:
            print(f"  模型参数量: {count_parameters(model):,}")

        acc, f1, y_true, y_pred, history = run_training(
            model, train_loader, val_loader, num_classes, device,
            fold=fold, output_dir=task_output_dir,
        )

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_histories.append(history)

        print(f"\n  Fold {fold} 结果: Accuracy={acc:.4f}, F1={f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))

        plot_confusion_matrix(y_true, y_pred, class_names, fold, task_output_dir)

    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold 交叉验证汇总 (Recording-level, 无数据泄露)")
    print(f"{'='*60}")
    print(f"  平均 Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"  平均 F1 Score: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
    print(f"  各折 Accuracy: {[f'{a:.4f}' for a in fold_accuracies]}")

    plot_confusion_matrix(all_y_true, all_y_pred, class_names, fold="all", output_dir=task_output_dir)
    plot_combined_training_curves(all_histories, task_output_dir)

    results = {
        "task": task,
        "model_type": model_type,
        "n_folds": n_folds,
        "split_level": "recording-level (no data leakage)",
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
    parser = argparse.ArgumentParser(description="Bonn EEG 癫痫检测 - PyTorch 训练脚本")
    parser.add_argument(
        "--task", type=str, default="binary",
        choices=["binary", "three", "five"],
        help="分类任务: binary(二分类), three(三分类), five(五分类)",
    )
    parser.add_argument(
        "--model", type=str, default="hybrid",
        choices=["hybrid", "bilstm", "cnn"],
        help="模型类型: hybrid(CNN+BiLSTM+Attention), bilstm(纯BiLSTM), cnn(纯1D-CNN)",
    )
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "kfold"],
        help="训练模式: single(单次分割快速验证), kfold(K折交叉验证)",
    )
    parser.add_argument(
        "--folds", type=int, default=K_FOLDS,
        help=f"K折交叉验证折数 (默认: {K_FOLDS})",
    )

    args = parser.parse_args()

    if args.mode == "kfold":
        train_kfold(task=args.task, model_type=args.model, n_folds=args.folds)
    else:
        train_single_split(task=args.task, model_type=args.model)


if __name__ == "__main__":
    main()
