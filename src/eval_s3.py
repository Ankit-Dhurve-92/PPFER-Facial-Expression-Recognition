import os
import sys
import glob
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_r3d
from loader import Cremad
from train_params import BASE_DIR

CODE_TO_NAME = {
    'ANG': 'Anger',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad'
}


def find_latest_s3_ckpt():
    preferred = os.path.join(BASE_DIR, "outputs/train-[2026-11-24]-[14:18]/s3_best_model.pth")
    if os.path.exists(preferred):
        return preferred
    matches = sorted(
        glob.glob(os.path.join(BASE_DIR, "outputs", "**", "s3_best_model.pth"), recursive=True),
        key=os.path.getmtime,
        reverse=True
    )
    if not matches:
        raise FileNotFoundError("s3_best_model.pth nahi mila!")
    return matches[0]


def infer_class_names(val_ds, num_classes=6):
    idx_to_code = {}
    for frames, labels in val_ds.data:
        emo_idx = int(labels[1])
        clip_folder = os.path.basename(os.path.dirname(frames[0]))
        parts = clip_folder.split('_')
        if len(parts) >= 3 and parts[2] in CODE_TO_NAME:
            idx_to_code[emo_idx] = parts[2]
        if len(idx_to_code) == num_classes:
            break

    default_codes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    names = []
    for i in range(num_classes):
        code = idx_to_code.get(i, default_codes[i])
        names.append(f"{CODE_TO_NAME.get(code, code)} ({code})")
    return names


def save_cm_png(cm, short_labels, overall_acc, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row Recall (%)')

        ax.set(
            xticks=np.arange(len(short_labels)),
            yticks=np.arange(len(short_labels)),
            xticklabels=short_labels,
            yticklabels=short_labels,
            ylabel='True Emotion',
            xlabel='Predicted Emotion',
            title=f'PPFER Stage-3 Confusion Matrix (CREMA-D Val Acc: {overall_acc:.2f}%)'
        )

        for i in range(len(short_labels)):
            for j in range(len(short_labels)):
                color = "white" if cm_norm[i, j] > 50 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)", ha="center", va="center", color=color, fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"PNG plot skipped ({e})", flush=True)
        return False


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_latest_s3_ckpt()
    save_dir = os.path.join(BASE_DIR, "outputs", "eval_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading Checkpoint: {ckpt_path}", flush=True)

    model = get_r3d().to(device)
    state = torch.load(ckpt_path, map_location=device)
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    model.eval()

    val_ds = Cremad('validation', 3)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
    )
    class_names = infer_class_names(val_ds, num_classes=6)
    short_labels = [c.split('(')[1].rstrip(')') for c in class_names]
    clip_names = [os.path.basename(os.path.dirname(frames[0])) for frames, _ in val_ds.data]

    all_preds, all_targets, all_confs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            videos = images.permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
            targets = labels[:, 1].numpy()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(videos)
                probs = torch.softmax(logits.float(), dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_confs.append(confs.cpu().numpy())
            all_targets.append(targets)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_conf = np.concatenate(all_confs)

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    class_acc = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0) * 100.0
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

    overall_acc = float((np.sum(tp) / np.sum(support)) * 100.0)
    macro_f1 = float(np.mean(f1) * 100.0)
    weighted_f1 = float((np.sum(f1 * support) / np.sum(support)) * 100.0)

    lines = []
    lines.append("=" * 78)
    lines.append("PPFER STAGE-3 EVALUATION REPORT (CREMA-D VALIDATION SET)")
    lines.append(f"Checkpoint : {ckpt_path}")
    lines.append(f"Total Clips: {len(y_true)}")
    lines.append("=" * 78)
    lines.append(f"Overall Accuracy : {overall_acc:.2f}%")
    lines.append(f"Macro F1-Score   : {macro_f1:.2f}%")
    lines.append(f"Weighted F1-Score: {weighted_f1:.2f}%")
    lines.append("-" * 78)
    lines.append(f"{'Class':<18} {'Support':>8} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1-Score':>10}")
    lines.append("-" * 78)
    for i, cname in enumerate(class_names):
        lines.append(
            f"{cname:<18} {support[i]:>8d} {class_acc[i]:>9.2f}% "
            f"{precision[i]*100:>10.2f}% {recall[i]*100:>8.2f}% {f1[i]*100:>9.2f}%"
        )
    lines.append("=" * 78)

    lines.append("\nCONFUSION MATRIX (Rows = True Label, Columns = Predicted Label):")
    header = f"{'True vs Pred':<14}" + "".join([f"{lbl:>9}" for lbl in short_labels])
    lines.append(header)
    lines.append("-" * len(header))
    for i, lbl in enumerate(short_labels):
        row_str = f"{lbl:<14}" + "".join([f"{cm[i, j]:>9d}" for j in range(num_classes)])
        lines.append(row_str)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # 1. Save text report
    report_file = os.path.join(save_dir, "s3_eval_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text + "\n")

    # 2. Save confusion matrix CSV
    cm_csv = os.path.join(save_dir, "s3_confusion_matrix.csv")
    with open(cm_csv, "w") as f:
        f.write("True/Pred," + ",".join(short_labels) + "\n")
        for i, lbl in enumerate(short_labels):
            f.write(lbl + "," + ",".join(map(str, cm[i])) + "\n")

    # 3. Save per-clip predictions CSV
    preds_csv = os.path.join(save_dir, "s3_per_clip_predictions.csv")
    with open(preds_csv, "w") as f:
        f.write("clip_name,true_label,pred_label,confidence_pct,is_correct\n")
        for cname, t, p, conf in zip(clip_names, y_true, y_pred, y_conf):
            f.write(f"{cname},{short_labels[t]},{short_labels[p]},{conf*100:.2f},{int(t == p)}\n")

    # 4. Save JSON metrics
    metrics_json = os.path.join(save_dir, "s3_metrics.json")
    summary_dict = {
        "checkpoint": ckpt_path,
        "total_clips": int(len(y_true)),
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            short_labels[i]: {
                "support": int(support[i]),
                "accuracy": round(float(class_acc[i]), 4),
                "precision": round(float(precision[i] * 100), 4),
                "recall": round(float(recall[i] * 100), 4),
                "f1_score": round(float(f1[i] * 100), 4),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist()
    }
    with open(metrics_json, "w") as f:
        json.dump(summary_dict, f, indent=2)

    # 5. Save PNG Heatmap
    cm_png = os.path.join(save_dir, "s3_confusion_matrix.png")
    png_saved = save_cm_png(cm, short_labels, overall_acc, cm_png)

    print(f"\nAll evaluation files saved inside: {save_dir}/")
    print("  - s3_eval_report.txt")
    print("  - s3_confusion_matrix.csv")
    print("  - s3_per_clip_predictions.csv")
    print("  - s3_metrics.json")
    if png_saved:
        print("  - s3_confusion_matrix.png")


if __name__ == '__main__':
    main()
