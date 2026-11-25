import os
import sys
import torch
from torchvision.utils import save_image
from torchvision.io import decode_image
from torchvision import transforms
from pytorch_wavelets import DWTForward, DWTInverse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import UNet
from train_params import s1_video_clip_dir, s2_video_clip_dir, image_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt_dir = "/home/ankit.dhurve/ppfer-main (2)/ppfer-main/outputs/train-[2026-11-22]-[23:32]"

f_hpr = UNet(3, 3).to(device).eval()
f_lpr = UNet(3, 3).to(device).eval()
f_hpr.load_state_dict(torch.load(os.path.join(ckpt_dir, "f_hpr_best.pth"), map_location=device, weights_only=True))
f_lpr.load_state_dict(torch.load(os.path.join(ckpt_dir, "f_lpr_best.pth"), map_location=device, weights_only=True))

dwt = DWTForward(J=1, wave='db1').to(device)
idwt = DWTInverse(wave='db1').to(device)

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
tf = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((image_size, image_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

os.makedirs(s2_video_clip_dir, exist_ok=True)
clip_names = sorted(os.listdir(s1_video_clip_dir))
exported, skipped = 0, 0

with torch.no_grad():
    for idx, cname in enumerate(clip_names):
        src_dir = os.path.join(s1_video_clip_dir, cname)
        if not os.path.isdir(src_dir):
            continue
        frame_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
        if not frame_files:
            continue
        if len(frame_files) > 16:
            step = len(frame_files) // 16
            frame_files = frame_files[::step][:16]

        dst_dir = os.path.join(s2_video_clip_dir, cname)
        if os.path.isdir(dst_dir) and len([f for f in os.listdir(dst_dir) if f.endswith(".jpg")]) == len(frame_files):
            skipped += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)
        imgs = torch.stack([decode_image(os.path.join(src_dir, f), mode='RGB') for f in frame_files])
        imgs = tf(imgs).to(device)

        low, high = dwt(imgs)
        high_band = high[0].mean(dim=2)
        filt_high = f_hpr(high_band)
        filt_high_3d = filt_high.unsqueeze(2).expand_as(high[0])
        filt_low = f_lpr(low)

        recon = idwt((filt_low, [filt_high_3d]))
        recon_denorm = torch.clamp(recon * std + mean, 0.0, 1.0)

        for f_i, img_t in enumerate(recon_denorm):
            save_image(img_t, os.path.join(dst_dir, f"{f_i+1:04d}.jpg"))
        exported += 1

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(clip_names):
            print(f"Progress {idx+1}/{len(clip_names)} | Skipped: {skipped} | Newly Exported: {exported}")

print(f"DONE! Total clips in s2_clips: {len(os.listdir(s2_video_clip_dir))}")
