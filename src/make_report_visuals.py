import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_params import BASE_DIR, s1_video_clip_dir, s2_video_clip_dir, s3_video_clip_dir
from loader import Cremad

CODE_TO_FULL = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}


def get_sample_triplets(target_codes=("HAP", "ANG", "DIS")):
    val_ds = Cremad("validation", 1)
    found = {}

    for frames, _ in val_ds.data:
        clip_folder = os.path.basename(os.path.dirname(frames[0]))
        parts = clip_folder.split("_")
        code = parts[2] if len(parts) >= 3 else ""
        if code in target_codes and code not in found:
            s1_dir = os.path.join(s1_video_clip_dir, clip_folder)
            s2_dir = os.path.join(s2_video_clip_dir, clip_folder)
            s3_dir = os.path.join(s3_video_clip_dir, clip_folder)

            if os.path.isdir(s1_dir) and os.path.isdir(s2_dir) and os.path.isdir(s3_dir):
                f1 = sorted([os.path.join(s1_dir, f) for f in os.listdir(s1_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                f2 = sorted([os.path.join(s2_dir, f) for f in os.listdir(s2_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                f3 = sorted([os.path.join(s3_dir, f) for f in os.listdir(s3_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                if f1 and f2 and f3:
                    mid = min(len(f1), len(f2), len(f3)) // 2
                    found[code] = (clip_folder, f1[mid], f2[mid], f3[mid])

        if len(found) == len(target_codes):
            break

    return [(code, *found[code]) for code in target_codes if code in found]


def generate_with_matplotlib(metrics, triplets, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = list(metrics["per_class"].keys())
    cm = np.array(metrics["confusion_matrix"], dtype=int)
    overall_acc = metrics["overall_accuracy"]
    macro_f1 = metrics["macro_f1"]

    # 1. Confusion Matrix Heatmap
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=10, fontweight="bold")

    ax.set(
        xticks=np.arange(len(codes)),
        yticks=np.arange(len(codes)),
        xticklabels=codes,
        yticklabels=codes,
        ylabel="True Emotion",
        xlabel="Predicted Emotion",
    )
    ax.set_title(f"PPFER Confusion Matrix (CREMA-D Val Acc: {overall_acc:.2f}%)", fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    for i in range(len(codes)):
        for j in range(len(codes)):
            text_color = "white" if cm_norm[i, j] > 50 else "#111111"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "report_confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # 2. Class-wise F1 Bar Chart
    sorted_items = sorted(metrics["per_class"].items(), key=lambda x: x[1]["f1_score"], reverse=True)
    labels = [f"{CODE_TO_FULL.get(k, k)} ({k})" for k, _ in sorted_items]
    f1_scores = [v["f1_score"] for _, v in sorted_items]
    colors = ["#1f77b4" if s >= macro_f1 else "#5b9bd5" for s in f1_scores]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    bars = ax.bar(labels, f1_scores, color=colors, width=0.55, edgecolor="#0f3d68", linewidth=0.8)
    ax.axhline(macro_f1, color="#d62728", linestyle="--", linewidth=1.5, label=f"Macro F1-Score: {macro_f1:.2f}%")

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.3, f"{score:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("F1-Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("Class-Wise Emotion Recognition F1-Score (CREMA-D)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "report_class_f1_chart.png"), dpi=300)
    plt.close(fig)

    # 3. Raw vs Stage-1 vs Stage-2 Visual Comparison
    if triplets:
        n_rows = len(triplets)
        fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3.1 * n_rows), dpi=300)
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        col_titles = ["Raw Face (Input)", "Stage-1: Anonymized", "Stage-2: Restored"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=10)

        for row, (code, clip_name, p1, p2, p3) in enumerate(triplets):
            for col, img_path in enumerate([p1, p2, p3]):
                img = Image.open(img_path).convert("RGB").resize((256, 256))
                axes[row, col].imshow(img)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor("#cccccc")
                    spine.set_linewidth(0.8)
            axes[row, 0].set_ylabel(f"{CODE_TO_FULL.get(code, code)} ({code})\n[{clip_name}]", fontsize=9.5, fontweight="bold", labelpad=8)

        plt.suptitle("Qualitative Privacy-Preserving Pipeline Comparison", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "report_stage_comparison.png"), dpi=300)
        plt.close(fig)


def generate_with_pil_fallback(metrics, triplets, save_dir):
    font = ImageFont.load_default()
    codes = list(metrics["per_class"].keys())
    cm = np.array(metrics["confusion_matrix"], dtype=int)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    # 1. Confusion Matrix via PIL
    cell = 95
    margin_l, margin_t = 110, 90
    w, h = margin_l + cell * 6 + 40, margin_t + cell * 6 + 50
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"PPFER Confusion Matrix (Val Acc: {metrics['overall_accuracy']:.2f}%)", fill="black", font=font)

    for j, c in enumerate(codes):
        draw.text((margin_l + j * cell + 35, margin_t - 25), c, fill="black", font=font)
    for i, c in enumerate(codes):
        draw.text((margin_l - 50, margin_t + i * cell + 40), c, fill="black", font=font)
        for j in range(6):
            val = cm_norm[i, j]
            shade = int(245 - (val / 100.0) * 190)
            fill_rgb = (shade, min(255, shade + 15), 255)
            x0, y0 = margin_l + j * cell, margin_t + i * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill_rgb, outline="#888888")
            txt_col = "white" if val > 55 else "black"
            draw.text((x0 + 22, y0 + 32), f"{cm[i, j]} ({val:.1f}%)", fill=txt_col, font=font)
    img.save(os.path.join(save_dir, "report_confusion_matrix.png"))

    # 2. Bar Chart via PIL
    sorted_items = sorted(metrics["per_class"].items(), key=lambda x: x[1]["f1_score"], reverse=True)
    bw, bh = 720, 420
    bimg = Image.new("RGB", (bw, bh), "white")
    bdraw = ImageDraw.Draw(bimg)
    bdraw.text((20, 15), f"Class-Wise F1-Score (Macro F1: {metrics['macro_f1']:.2f}%)", fill="black", font=font)
    for idx, (k, v) in enumerate(sorted_items):
        score = v["f1_score"]
        y0 = 60 + idx * 52
        bar_len = int((score / 100.0) * 450)
        bdraw.text((20, y0 + 10), f"{CODE_TO_FULL.get(k, k)} ({k})", fill="black", font=font)
        bdraw.rectangle([140, y0, 140 + bar_len, y0 + 32], fill=(31, 119, 180))
        bdraw.text((150 + bar_len, y0 + 10), f"{score:.2f}%", fill="black", font=font)
    bimg.save(os.path.join(save_dir, "report_class_f1_chart.png"))

    # 3. Sample Triplets via PIL
    if triplets:
        tile = 256
        pad = 20
        gw = 140 + 3 * (tile + pad)
        gh = 60 + len(triplets) * (tile + pad)
        gimg = Image.new("RGB", (gw, gh), "white")
        gdraw = ImageDraw.Draw(gimg)
        headers = ["Raw Face (Input)", "Stage-1: Anonymized", "Stage-2: Restored"]
        for col, htxt in enumerate(headers):
            gdraw.text((140 + col * (tile + pad) + 50, 20), htxt, fill="black", font=font)
        for row, (code, clip_name, p1, p2, p3) in enumerate(triplets):
            y0 = 50 + row * (tile + pad)
            gdraw.text((15, y0 + 110), f"{CODE_TO_FULL.get(code, code)} ({code})", fill="black", font=font)
            for col, p in enumerate([p1, p2, p3]):
                im = Image.open(p).convert("RGB").resize((tile, tile))
                gimg.paste(im, (140 + col * (tile + pad), y0))
        gimg.save(os.path.join(save_dir, "report_stage_comparison.png"))


def main():
    save_dir = os.path.join(BASE_DIR, "outputs", "eval_results")
    metrics_path = os.path.join(save_dir, "s3_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"s3_metrics.json nahi mila: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    triplets = get_sample_triplets(("HAP", "ANG", "DIS"))
    print(f"Found {len(triplets)} sample clip triplets for visual comparison.")

    try:
        generate_with_matplotlib(metrics, triplets, save_dir)
        print("Generated 300-DPI report charts using Matplotlib!")
    except ImportError:
        generate_with_pil_fallback(metrics, triplets, save_dir)
        print("Matplotlib unavailable; generated report charts using PIL fallback!")

    print("\nSaved files:")
    for fn in ["report_confusion_matrix.png", "report_class_f1_chart.png", "report_stage_comparison.png"]:
        fp = os.path.join(save_dir, fn)
        if os.path.exists(fp):
            print(f"  - {fp} ({os.path.getsize(fp)//1024} KB)")


if __name__ == "__main__":
    main()
