"""
Bonn EEG 癫痫检测 - 独立推理脚本

加载训练好的模型权重，对单个 EEG 文件或目录进行分类预测。
输出分类结果、置信度，以及 attention 热力图（仅限 hybrid/bilstm 模型）。
"""
import os
import glob
import argparse
import numpy as np
import torch

from config import (
    WINDOW_SIZE, WINDOW_STRIDE, POINTS_PER_FILE,
    BINARY_CLASS_NAMES, THREE_CLASS_NAMES, FIVE_CLASS_NAMES,
    OUTPUT_DIR,
)
from data_loader import load_single_file, sliding_window_segment
from model import build_model
from sklearn.preprocessing import StandardScaler


TASK_CLASS_NAMES = {
    "binary": BINARY_CLASS_NAMES,
    "three": THREE_CLASS_NAMES,
    "five": FIVE_CLASS_NAMES,
}


def predict_file(model, filepath: str, task: str, device: torch.device,
                 scaler: StandardScaler | None = None):
    """
    Run inference on a single EEG file.

    Returns dict with keys: filepath, predicted_class, confidence, all_probs,
    segment_preds, attn_weights (list or None).
    """
    signal = load_single_file(filepath)
    if len(signal) != POINTS_PER_FILE:
        raise ValueError(f"Expected {POINTS_PER_FILE} points, got {len(signal)} in {filepath}")

    segments = sliding_window_segment(signal, WINDOW_SIZE, WINDOW_STRIDE)

    if scaler is not None:
        orig_shape = segments.shape
        segments = scaler.transform(segments.reshape(-1, 1)).reshape(orig_shape)

    X = torch.from_numpy(segments[..., np.newaxis].astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(X)

    class_names = TASK_CLASS_NAMES[task]
    num_classes = len(class_names)

    if num_classes == 2:
        probs = torch.sigmoid(logits).cpu().numpy()
        seg_preds = (probs > 0.5).astype(int)
        vote_counts = np.bincount(seg_preds, minlength=2)
        predicted_class_idx = int(np.argmax(vote_counts))
        confidence = float(vote_counts[predicted_class_idx] / len(seg_preds))
        all_probs = [float(1 - probs.mean()), float(probs.mean())]
    else:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        seg_preds = probs.argmax(axis=1)
        mean_probs = probs.mean(axis=0)
        predicted_class_idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[predicted_class_idx])
        all_probs = mean_probs.tolist()

    attn_np = None
    if attn_weights is not None:
        attn_np = attn_weights.cpu().numpy()

    return {
        "filepath": filepath,
        "predicted_class": class_names[predicted_class_idx],
        "predicted_class_idx": predicted_class_idx,
        "confidence": confidence,
        "all_probs": {name: prob for name, prob in zip(class_names, all_probs)},
        "num_segments": len(seg_preds),
        "segment_preds": seg_preds.tolist(),
        "attn_weights": attn_np,
        "signal": signal,
    }


def main():
    parser = argparse.ArgumentParser(description="Bonn EEG 癫痫检测 - 推理脚本")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="训练好的模型权重文件路径 (.pt)",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入 EEG 文件路径或包含 .txt 文件的目录",
    )
    parser.add_argument(
        "--task", type=str, default="binary",
        choices=["binary", "three", "five"],
        help="分类任务",
    )
    parser.add_argument(
        "--model", type=str, default="hybrid",
        choices=["hybrid", "bilstm", "cnn"],
        help="模型架构（必须与训练时一致）",
    )
    parser.add_argument(
        "--attention_viz", action="store_true",
        help="是否保存 attention 热力图",
    )

    args = parser.parse_args()

    class_names = TASK_CLASS_NAMES[args.task]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    model = build_model(args.model, num_classes=num_classes).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] 模型已加载: {args.model_path}")

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.txt")))
        if not files:
            print(f"[ERROR] 目录 {args.input} 中未找到 .txt 文件")
            return
    else:
        files = [args.input]

    print(f"[INFO] 待预测文件: {len(files)} 个\n")

    viz_dir = None
    if args.attention_viz and args.model != "cnn":
        viz_dir = os.path.join(OUTPUT_DIR, "attention_viz")
        os.makedirs(viz_dir, exist_ok=True)

    for fpath in files:
        try:
            result = predict_file(model, fpath, args.task, device)
        except Exception as e:
            print(f"  [SKIP] {fpath}: {e}")
            continue

        fname = os.path.basename(fpath)
        print(f"  {fname}: {result['predicted_class']} "
              f"(confidence={result['confidence']:.2%})")
        for cname, prob in result["all_probs"].items():
            print(f"    {cname}: {prob:.4f}")

        if viz_dir and result["attn_weights"] is not None:
            from visualize import plot_attention_heatmap
            seg_idx = 0
            seg_signal = result["signal"][:WINDOW_SIZE]
            attn = result["attn_weights"][seg_idx]
            out_path = os.path.join(viz_dir, f"attention_{fname.replace('.txt', '.png')}")
            plot_attention_heatmap(
                signal=seg_signal,
                attn_weights=attn,
                label="unknown",
                pred_label=result["predicted_class"],
                confidence=result["confidence"],
                output_path=out_path,
            )
            print(f"    Attention 热力图已保存: {out_path}")

    print("\n[INFO] 推理完成")


if __name__ == "__main__":
    main()
