import os
import sys
import glob
import json
import inspect
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models
from loader import Cremad
from train_params import BASE_DIR


def extract_state_dict(ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break
    return {k.replace("module.", ""): v for k, v in raw.items()}


def infer_num_classes(state_dict):
    candidates = [
        v.shape[0] for k, v in state_dict.items()
        if isinstance(v, torch.Tensor) and v.ndim == 2 and any(t in k for t in ["fc", "head", "classifier", "linear"])
    ]
    return int(candidates[-1]) if candidates else 91


def build_identity_model(ckpt_path, device):
    state_dict = extract_state_dict(ckpt_path)
    num_classes = infer_num_classes(state_dict)
    print(f"Checkpoint keys: {len(state_dict)} | Inferred Identity Classes: {num_classes}", flush=True)

    candidates = []

    # 1. Collect all builders from src/models.py
    for name, obj in inspect.getmembers(models):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            candidates.append((f"models.{name}", obj))

    # 2. Add standard torchvision ResNet backbones as fallback
    try:
        import torchvision.models as tvm
        for rname in ["resnet50", "resnet34", "resnet18", "resnext50_32x4d"]:
            if hasattr(tvm, rname):
                candidates.append((f"torchvision.{rname}", getattr(tvm, rname)))
    except Exception:
        pass

    for cname, builder in candidates:
        for kwargs in [{"num_classes": num_classes}, {}]:
            try:
                net = builder(**kwargs)
                if not isinstance(net, nn.Module):
                    continue
                net.load_state_dict(state_dict, strict=True)
                print(f"Matched Identity Controller Architecture: {cname}({kwargs})", flush=True)
                return net.to(device).eval(), num_classes
            except Exception:
                continue

    # Relaxed match check if strict=True didn't match (e.g., extra buffer keys)
    for cname, builder in candidates:
        for kwargs in [{"num_classes": num_classes}, {}]:
            try:
                net = builder(**kwargs)
                if not isinstance(net, nn.Module):
                    continue
                net_keys = set(net.state_dict().keys())
                ckpt_keys = set(state_dict.keys())
                overlap = len(net_keys & ckpt_keys) / max(len(ckpt_keys), 1)
                if overlap > 0.90:
                    net.load_state_dict(state_dict, strict=False)
                    print(f"Matched Identity Controller (overlap {overlap*100:.1f}%): {cname}({kwargs})", flush=True)
                    return net.to(device).eval(), num_classes
            except Exception:
                continue

    print("Sample checkpoint keys:", list(state_dict.keys())[:10])
    raise RuntimeError("Could not auto-match c_hpr_identity.pth architecture. Check src/ for controller definition.")


def is_3d_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            return True
        if isinstance(m, nn.Conv2d):
            return False
    return False


def evaluate_stage(model, stage, device, num_classes, use_3d):
    ds = Cremad('validation', stage)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
    )

    clip_top1_correct = 0
    clip_top5_correct = 0
    frame_top1_correct = 0
    total_clips = 0
    total_frames = 0

    with torch.no_grad():
        for images, labels in loader:
            # labels[:, 0] = Actor/Identity ID, labels[:, 1] = Emotion ID
            id_targets = labels[:, 0].long().to(device, non_blocking=True)
            B, T, C, H, W = images.shape

            if use_3d:
                x = images.permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    clip_logits = model(x).float()
                clip_probs = torch.softmax(clip_logits, dim=1)
                frame_top1_correct += (clip_probs.argmax(dim=1) == id_targets).sum().item() * T
                total_frames += B * T
            else:
                x = images.reshape(B * T, C, H, W).contiguous().to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    frame_logits = model(x).float()
                frame_probs = torch.softmax(frame_logits, dim=1).view(B, T, -1)

                # Frame-level accuracy
                expanded_targets = id_targets.unsqueeze(1).expand(B, T)
                frame_top1_correct += (frame_probs.argmax(dim=2) == expanded_targets).sum().item()
                total_frames += B * T

                # Clip-level accuracy (mean probability across 16 frames)
                clip_probs = frame_probs.mean(dim=1)

            clip_top1_correct += (clip_probs.argmax(dim=1) == id_targets).sum().item()
            k = min(5, clip_probs.shape[1])
            top5_preds = clip_probs.topk(k, dim=1).indices
            clip_top5_correct += (top5_preds == id_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_clips += B

    return {
        "total_clips": total_clips,
        "clip_top1_acc": (clip_top1_correct / max(total_clips, 1)) * 100.0,
        "clip_top5_acc": (clip_top5_correct / max(total_clips, 1)) * 100.0,
        "frame_top1_acc": (frame_top1_correct / max(total_frames, 1)) * 100.0,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(BASE_DIR, "pretrained_weights", "c_hpr_identity.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Controller not found: {ckpt_path}")

    print(f"Loading Identity Controller: {ckpt_path}", flush=True)
    model, num_classes = build_identity_model(ckpt_path, device)
    use_3d = is_3d_model(model)
    print(f"Model Input Mode: {'3D Video (B,C,T,H,W)' if use_3d else '2D Frame-wise (B*T,C,H,W)'}\n", flush=True)

    stage_names = {
        1: "Raw Face Clips (Unprotected)",
        2: "Stage-1 Output (s2_clips - Anonymized)",
        3: "Stage-2 Output (s3_clips - Final Restored)",
    }

    results = {}
    for st in [1, 2, 3]:
        print(f"Evaluating Stage {st}: {stage_names[st]}...", flush=True)
        res = evaluate_stage(model, st, device, num_classes, use_3d)
        results[st] = res

    raw_acc = results[1]["clip_top1_acc"]
    s2_drop = raw_acc - results[2]["clip_top1_acc"]
    s3_drop = raw_acc - results[3]["clip_top1_acc"]

    lines = []
    lines.append("=" * 84)
    lines.append("PPFER PRIVACY PROTECTION EVALUATION (IDENTITY LEAKAGE TEST)")
    lines.append(f"Identity Controller: {ckpt_path}")
    lines.append(f"Validation Clips   : {results[1]['total_clips']} (91 CREMA-D Actors)")
    lines.append("=" * 84)
    lines.append(f"{'Dataset Stage':<42} {'Clip Top-1':>11} {'Clip Top-5':>11} {'Frame Top-1':>12}")
    lines.append("-" * 84)
    for st in [1, 2, 3]:
        r = results[st]
        lines.append(
            f"{stage_names[st]:<42} {r['clip_top1_acc']:>10.2f}% {r['clip_top5_acc']:>10.2f}% {r['frame_top1_acc']:>11.2f}%"
        )
    lines.append("=" * 84)
    lines.append(f"Identity Accuracy Drop after Stage 1 (s2_clips): -{s2_drop:.2f}% (Higher drop = Better privacy)")
    lines.append(f"Identity Accuracy Drop after Stage 2 (s3_clips): -{s3_drop:.2f}% (Final system privacy gain)")
    lines.append("=" * 84)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    save_dir = os.path.join(BASE_DIR, "outputs", "eval_results")
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, "privacy_eval_report.txt")
    json_path = os.path.join(save_dir, "privacy_metrics.json")

    with open(txt_path, "w") as f:
        f.write(report_text + "\n")

    with open(json_path, "w") as f:
        json.dump({
            "controller": ckpt_path,
            "stages": {stage_names[st]: results[st] for st in [1, 2, 3]},
            "privacy_gain_clip_top1_drop": {
                "after_stage1_s2_clips": round(s2_drop, 4),
                "after_stage2_s3_clips": round(s3_drop, 4),
            }
        }, f, indent=2)

    print(f"\nPrivacy results saved to:\n  - {txt_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
