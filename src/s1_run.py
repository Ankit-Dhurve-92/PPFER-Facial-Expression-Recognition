import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.io import decode_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_wavelets import DWTForward, DWTInverse

from train_params import (
    batch_size, learning_rate, num_epochs, num_workers, gpu_ids,
    log_path, tensorboard_log_path, output_path_time,
    c_lpr_pretrain_weight, s1_video_clip_dir, s2_video_clip_dir, image_size
)
from models import get_unet, get_controller
from loader import Cremad


class Solver:
    def __init__(self) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.lr = learning_rate
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.dwt = DWTForward(J=1, wave='db1').to(self.device)
        self.idwt = DWTInverse(wave='db1').to(self.device)

        self.f_hpr = nn.DataParallel(get_unet().to(self.device), device_ids=self.gpu_ids)
        self.f_lpr = nn.DataParallel(get_unet().to(self.device), device_ids=self.gpu_ids)

        self.c_hpr = nn.DataParallel(get_controller(91).to(self.device), device_ids=self.gpu_ids)
        self.c_lpr = nn.DataParallel(get_controller(6, c_lpr_pretrain_weight).to(self.device), device_ids=self.gpu_ids)
        for p in self.c_hpr.parameters():
            p.requires_grad = False
        for p in self.c_lpr.parameters():
            p.requires_grad = False

        self.optimizer_hpr = torch.optim.Adam(self.f_hpr.parameters(), lr=self.lr)
        self.optimizer_lpr = torch.optim.Adam(self.f_lpr.parameters(), lr=self.lr)

        self.train_ds = Cremad('train', 1)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        self.val_ds = Cremad('validation', 1)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        self.output_path_time = output_path_time
        self.tensorboard_log_path = tensorboard_log_path

    def _adv_privacy_loss(self, logits, orig_band, filt_band, rec_weight=2.0):
        log_probs = F.log_softmax(logits, dim=1)
        num_classes = logits.size(1)
        uniform_target = torch.full_like(log_probs, 1.0 / num_classes)
        uniform_kl = F.kl_div(log_probs, uniform_target, reduction='batchmean')
        rec_loss = F.l1_loss(filt_band, orig_band)
        return uniform_kl + rec_weight * rec_loss

    def train(self, epoch: int, writer: SummaryWriter):
        losses_h, losses_l, acc_h_list, acc_l_list = [], [], [], []
        self.f_hpr.train()
        self.f_lpr.train()
        self.c_hpr.eval()
        self.c_lpr.eval()

        for i, (imgs, labels) in enumerate(self.train_loader):
            B, T, C, H, W = imgs.shape
            imgs_flat = imgs.view(B * T, C, H, W).to(self.device)
            id_labels = labels[:, 0].to(self.device)
            emo_labels = labels[:, 1].to(self.device)

            with torch.no_grad():
                low, high = self.dwt(imgs_flat)
                high_band = high[0].mean(dim=2)
                del imgs_flat, high

            # 1. High-frequency privacy filtering (FP32 + grad clip)
            self.optimizer_hpr.zero_grad()
            filt_high = self.f_hpr(high_band)
            high_logits = self.c_hpr(filt_high).view(B, T, -1).mean(dim=1)
            high_loss = self._adv_privacy_loss(high_logits, high_band, filt_high, rec_weight=2.5)
            high_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.f_hpr.parameters(), max_norm=1.0)
            self.optimizer_hpr.step()

            high_acc = (high_logits.argmax(1) == id_labels).float().mean().item() * 100.0
            h_loss_val = high_loss.item()
            del filt_high, high_logits, high_loss, high_band

            # 2. Low-frequency privacy filtering (FP32 + grad clip)
            self.optimizer_lpr.zero_grad()
            filt_low = self.f_lpr(low)
            low_logits = self.c_lpr(filt_low).view(B, T, -1).mean(dim=1)
            low_loss = self._adv_privacy_loss(low_logits, low, filt_low, rec_weight=3.0)
            low_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.f_lpr.parameters(), max_norm=1.0)
            self.optimizer_lpr.step()

            low_acc = (low_logits.argmax(1) == emo_labels).float().mean().item() * 100.0
            l_loss_val = low_loss.item()
            del filt_low, low_logits, low_loss, low

            losses_h.append(h_loss_val)
            losses_l.append(l_loss_val)
            acc_h_list.append(high_acc)
            acc_l_list.append(low_acc)

            if i % 150 == 0:
                msg = (
                    f"S1 Train Epoch {epoch+1}/{self.num_epochs} | Step {i}/{len(self.train_loader)} | "
                    f"H-Loss: {h_loss_val:.4f} (ID Acc: {high_acc:.1f}%) | "
                    f"L-Loss: {l_loss_val:.4f} (Emo Acc: {low_acc:.1f}%)"
                )
                print(msg)
                logging.info(msg)

        return (
            sum(losses_h) / len(losses_h), sum(losses_l) / len(losses_l),
            sum(acc_h_list) / len(acc_h_list), sum(acc_l_list) / len(acc_l_list)
        )

    def eval(self, epoch: int, writer: SummaryWriter):
        losses_h, losses_l, acc_h_list, acc_l_list = [], [], [], []
        self.f_hpr.eval()
        self.f_lpr.eval()
        self.c_hpr.eval()
        self.c_lpr.eval()

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                B, T, C, H, W = imgs.shape
                imgs_flat = imgs.view(B * T, C, H, W).to(self.device)
                id_labels = labels[:, 0].to(self.device)
                emo_labels = labels[:, 1].to(self.device)

                low, high = self.dwt(imgs_flat)
                high_band = high[0].mean(dim=2)

                filt_high = self.f_hpr(high_band)
                high_logits = self.c_hpr(filt_high).view(B, T, -1).mean(dim=1)
                high_loss = self._adv_privacy_loss(high_logits, high_band, filt_high, rec_weight=2.5)

                filt_low = self.f_lpr(low)
                low_logits = self.c_lpr(filt_low).view(B, T, -1).mean(dim=1)
                low_loss = self._adv_privacy_loss(low_logits, low, filt_low, rec_weight=3.0)

                high_acc = (high_logits.argmax(1) == id_labels).float().mean().item() * 100.0
                low_acc = (low_logits.argmax(1) == emo_labels).float().mean().item() * 100.0
                losses_h.append(high_loss.item())
                losses_l.append(low_loss.item())
                acc_h_list.append(high_acc)
                acc_l_list.append(low_acc)

        val_lh = sum(losses_h) / len(losses_h)
        val_ll = sum(losses_l) / len(losses_l)
        val_ah = sum(acc_h_list) / len(acc_h_list)
        val_al = sum(acc_l_list) / len(acc_l_list)

        writer.add_scalar('S1_Val/High_Loss', val_lh, epoch)
        writer.add_scalar('S1_Val/Low_Loss', val_ll, epoch)
        writer.add_scalar('S1_Val/ID_Privacy_Acc', val_ah, epoch)
        writer.add_scalar('S1_Val/Low_Emo_Acc', val_al, epoch)
        return val_lh, val_ll, val_ah, val_al

    def export_stage2_clips(self):
        print("\n=== Exporting Stage-1 Anonymized Clips for Stage 2 ===")
        logging.info("Exporting Stage-1 Anonymized Clips for Stage 2...")
        os.makedirs(s2_video_clip_dir, exist_ok=True)
        self.f_hpr.eval()
        self.f_lpr.eval()

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tf = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        clip_names = sorted(os.listdir(s1_video_clip_dir))
        with torch.no_grad():
            for idx, cname in enumerate(clip_names):
                src_dir = os.path.join(s1_video_clip_dir, cname)
                if not os.path.isdir(src_dir):
                    continue
                dst_dir = os.path.join(s2_video_clip_dir, cname)
                os.makedirs(dst_dir, exist_ok=True)
                frame_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
                if not frame_files:
                    continue
                if len(frame_files) > 16:
                    step = len(frame_files) // 16
                    frame_files = frame_files[::step][:16]

                imgs = torch.stack([decode_image(os.path.join(src_dir, f), mode='RGB') for f in frame_files])
                imgs = tf(imgs).to(self.device)

                low, high = self.dwt(imgs)
                high_band = high[0].mean(dim=2)
                filt_high = self.f_hpr(high_band)
                filt_high_3d = filt_high.unsqueeze(2).expand_as(high[0])
                filt_low = self.f_lpr(low)

                recon = self.idwt((filt_low, [filt_high_3d]))
                recon_denorm = torch.clamp(recon * std + mean, 0.0, 1.0)

                for f_i, img_t in enumerate(recon_denorm):
                    save_image(img_t, os.path.join(dst_dir, f"{f_i+1:04d}.jpg"))

                if (idx + 1) % 1000 == 0 or (idx + 1) == len(clip_names):
                    print(f"Exported {idx+1}/{len(clip_names)} clips to {s2_video_clip_dir}")

    def run(self):
        writer = SummaryWriter(log_dir=self.tensorboard_log_path)
        best_h_loss = float('inf')
        best_l_loss = float('inf')
        h_ckpt = os.path.join(self.output_path_time, 'f_hpr_best.pth')
        l_ckpt = os.path.join(self.output_path_time, 'f_lpr_best.pth')

        for epoch in range(self.num_epochs):
            tr_lh, tr_ll, tr_ah, tr_al = self.train(epoch, writer)
            va_lh, va_ll, va_ah, va_al = self.eval(epoch, writer)

            summary = (
                f"Epoch {epoch+1}/{self.num_epochs} SUMMARY | "
                f"Val High Loss: {va_lh:.4f} (ID Acc: {va_ah:.2f}%) | "
                f"Val Low Loss: {va_ll:.4f} (Emo Acc: {va_al:.2f}%)"
            )
            print(summary)
            logging.info(summary)

            if not math.isnan(va_lh) and va_lh < best_h_loss:
                best_h_loss = va_lh
                torch.save(self.f_hpr.module.state_dict(), h_ckpt)
            if not math.isnan(va_ll) and va_ll < best_l_loss:
                best_l_loss = va_ll
                torch.save(self.f_lpr.module.state_dict(), l_ckpt)

        writer.close()

        self.f_hpr.module.load_state_dict(torch.load(h_ckpt))
        self.f_lpr.module.load_state_dict(torch.load(l_ckpt))
        self.export_stage2_clips()
        print("Stage 1 Training & Export 100% Completed!")
        logging.info("Stage 1 Training & Export 100% Completed!")


if __name__ == "__main__":
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s: %(message)s')
    solver = Solver()
    solver.run()
