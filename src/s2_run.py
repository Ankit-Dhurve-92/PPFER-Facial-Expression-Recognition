import os
import sys
import glob
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image
from torchvision.io import decode_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
uvit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'U-ViT'))
if uvit_path not in sys.path:
    sys.path.insert(0, uvit_path)

from libs.uvit import UViT
from uvit.autoencoder import get_model as get_autoencoder
from models import get_feature_controller
from loader import Cremad
from train_params import (
    BASE_DIR, image_size,
    s2_video_clip_dir, s3_video_clip_dir,
    uvit_256_weight, log_path, tensorboard_log_path, output_path_time
)


class LatentCompensator(nn.Module):
    def __init__(self, autoencoder, uvit, controller):
        super().__init__()
        self.autoencoder = autoencoder
        self.uvit = uvit
        self.controller = controller
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _decode_and_eval_chunk(self, z_chunk):
        recon_m11 = self.autoencoder.decode(z_chunk)
        recon_01 = torch.clamp((recon_m11 + 1.0) * 0.5, 0.0, 1.0)
        recon_norm = (recon_01 - self.mean) / self.std
        logits = self.controller(recon_norm)
        return logits, recon_01

    def forward(self, x_01, return_image_only=False):
        x_m11 = x_01 * 2.0 - 1.0
        B_flat = x_m11.size(0)
        with torch.no_grad():
            z = torch.cat([self.autoencoder.encode(x_m11[s:s+4]) for s in range(0, B_flat, 4)], dim=0)

        t = torch.full((B_flat,), 100, dtype=torch.long, device=z.device)
        y = torch.full((B_flat,), 1000, dtype=torch.long, device=z.device)

        z_comp = z + self.uvit(z, t, y)

        if return_image_only:
            out_chunks = []
            for s in range(0, B_flat, 4):
                r_m11 = self.autoencoder.decode(z_comp[s:s+4])
                out_chunks.append(torch.clamp((r_m11 + 1.0) * 0.5, 0.0, 1.0))
            return torch.cat(out_chunks, dim=0)

        logits_list, recon_list = [], []
        for s in range(0, B_flat, 4):
            z_c = z_comp[s:s+4]
            if self.training and z_c.requires_grad:
                lg, rc = checkpoint(self._decode_and_eval_chunk, z_c, use_reentrant=False)
            else:
                lg, rc = self._decode_and_eval_chunk(z_c)
            logits_list.append(lg)
            recon_list.append(rc)

        return torch.cat(logits_list, dim=0), torch.cat(recon_list, dim=0)


class S2Solver:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.num_epochs = 1
        self.batch_size = 1
        self.accum_steps = 2
        self.num_workers = 4

        vae_path = os.path.join(BASE_DIR, "pretrained_weights/autoencoder_kl.pth")
        autoencoder = get_autoencoder(vae_path)
        for p in autoencoder.parameters():
            p.requires_grad = False
        autoencoder.eval()

        uvit = UViT(
            img_size=image_size // 8, patch_size=2, in_chans=4,
            embed_dim=1024, depth=20, num_heads=16, num_classes=1001,
            conv=False, use_checkpoint=True
        )

        # Resume from previously saved best Stage-2 checkpoint if available
        prev_ckpts = sorted(
            glob.glob(os.path.join(BASE_DIR, '**', 's2_best_model.pth'), recursive=True),
            key=os.path.getmtime, reverse=True
        )
        if prev_ckpts:
            state = torch.load(prev_ckpts[0], map_location='cpu', weights_only=True)
            uvit.load_state_dict(state, strict=True)
            if self.rank == 0:
                print(f"Resumed U-ViT weights from: {prev_ckpts[0]}", flush=True)
        elif os.path.exists(uvit_256_weight):
            state = torch.load(uvit_256_weight, map_location='cpu', weights_only=True)
            clean_state = {k.replace('module.', ''): v for k, v in state.items()}
            uvit.load_state_dict(clean_state, strict=False)

        controller = get_feature_controller()
        for p in controller.parameters():
            p.requires_grad = False
        controller.eval()

        base_net = LatentCompensator(autoencoder, uvit, controller).to(self.device)
        self.net = DDP(base_net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        self.optimizer = torch.optim.AdamW(self.net.module.uvit.parameters(), lr=4e-5, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda')
        self.ce_loss = nn.CrossEntropyLoss()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.train_ds = Cremad('train', 2)
        self.train_sampler = DistributedSampler(self.train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, sampler=self.train_sampler,
            num_workers=self.num_workers, pin_memory=True
        )

        self.val_ds = Cremad('validation', 2)
        self.val_sampler = DistributedSampler(self.val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, sampler=self.val_sampler,
            num_workers=self.num_workers, pin_memory=True
        )

        self.output_path_time = output_path_time
        self.tensorboard_log_path = tensorboard_log_path
        if self.rank == 0:
            print(f"Dual-GPU DDP Active ({world_size} GPUs) | Steps/Epoch per GPU: {len(self.train_loader)}", flush=True)

    def train(self, epoch: int):
        losses, accs = [], []
        self.train_sampler.set_epoch(epoch)
        self.net.train()
        self.net.module.autoencoder.eval()
        self.net.module.controller.eval()
        self.optimizer.zero_grad()

        for i, (images, labels) in enumerate(self.train_loader):
            B, T, C, H, W = images.shape
            imgs_norm = images.view(B * T, C, H, W).to(self.device, non_blocking=True)
            imgs_01 = torch.clamp(imgs_norm * self.std + self.mean, 0.0, 1.0)
            emo_labels = labels[:, 1].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, recon_01 = self.net(imgs_01)
                clip_logits = logits.view(B, T, -1).mean(dim=1)
                emo_l = self.ce_loss(clip_logits, emo_labels)
                rec_l = F.l1_loss(recon_01, imgs_01)
                loss = (emo_l + 1.5 * rec_l) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accum_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.module.uvit.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            full_loss = loss.item() * self.accum_steps
            batch_acc = (clip_logits.argmax(1) == emo_labels).float().mean().item() * 100.0
            losses.append(full_loss)
            accs.append(batch_acc)

            if self.rank == 0 and i % 25 == 0:
                avg_l = sum(losses[-50:]) / len(losses[-50:])
                avg_a = sum(accs[-50:]) / len(accs[-50:])
                msg = (
                    f"S2-DDP Epoch {epoch+1}/{self.num_epochs} | Step {i}/{len(self.train_loader)} | "
                    f"Loss: {full_loss:.4f} (Avg: {avg_l:.4f} | Emo: {emo_l.item():.4f}, Rec: {rec_l.item():.4f}) | "
                    f"Emo Acc: {avg_a:.1f}%"
                )
                print(msg, flush=True)
                logging.info(msg)

        stats = torch.tensor([sum(losses), len(losses), sum(accs), len(accs)], device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (stats[0] / stats[1]).item(), (stats[2] / stats[3]).item()

    def evaluation(self, epoch: int):
        losses, accs = [], []
        self.net.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                B, T, C, H, W = images.shape
                imgs_norm = images.view(B * T, C, H, W).to(self.device, non_blocking=True)
                imgs_01 = torch.clamp(imgs_norm * self.std + self.mean, 0.0, 1.0)
                emo_labels = labels[:, 1].to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, recon_01 = self.net(imgs_01)
                    clip_logits = logits.view(B, T, -1).mean(dim=1)
                    emo_l = self.ce_loss(clip_logits, emo_labels)
                    rec_l = F.l1_loss(recon_01, imgs_01)
                    loss = emo_l + 1.5 * rec_l

                batch_acc = (clip_logits.argmax(1) == emo_labels).float().mean().item() * 100.0
                losses.append(loss.item())
                accs.append(batch_acc)

        stats = torch.tensor([sum(losses), len(losses), sum(accs), len(accs)], device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (stats[0] / stats[1]).item(), (stats[2] / stats[3]).item()

    def export_stage3_clips(self):
        if self.rank == 0:
            print("\n=== Exporting Stage-2 Compensated Clips on 2 GPUs in Parallel ===", flush=True)
            logging.info("Exporting Stage-2 Compensated Clips on 2 GPUs...")
        os.makedirs(s3_video_clip_dir, exist_ok=True)
        self.net.eval()

        tf = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((image_size, image_size)),
        ])

        clip_names = sorted(os.listdir(s2_video_clip_dir))
        with torch.no_grad():
            for idx in range(self.rank, len(clip_names), self.world_size):
                cname = clip_names[idx]
                src_dir = os.path.join(s2_video_clip_dir, cname)
                if not os.path.isdir(src_dir):
                    continue
                dst_dir = os.path.join(s3_video_clip_dir, cname)
                os.makedirs(dst_dir, exist_ok=True)
                existing = [f for f in os.listdir(dst_dir) if f.endswith(".jpg")]
                if len(existing) >= 16:
                    continue

                frame_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
                if not frame_files:
                    continue
                if len(frame_files) > 16:
                    step = len(frame_files) // 16
                    frame_files = frame_files[::step][:16]

                imgs_01 = torch.stack([decode_image(os.path.join(src_dir, f), mode='RGB') for f in frame_files])
                imgs_01 = tf(imgs_01).to(self.device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    comp_01 = self.net.module(imgs_01, return_image_only=True)

                for f_i, img_t in enumerate(comp_01.float()):
                    save_image(img_t, os.path.join(dst_dir, f"{f_i+1:04d}.jpg"))

                if self.rank == 0 and ((idx + 1) % 500 < self.world_size or idx + self.world_size >= len(clip_names)):
                    print(f"Exported ~{min(idx+self.world_size, len(clip_names))}/{len(clip_names)} clips to {s3_video_clip_dir}", flush=True)

        dist.barrier()

    def run(self):
        writer = SummaryWriter(log_dir=self.tensorboard_log_path) if self.rank == 0 else None
        best_val_loss = 1.0053
        os.makedirs(self.output_path_time, exist_ok=True)
        best_ckpt = os.path.join(self.output_path_time, 's2_best_model.pth')
        import shutil; prev = sorted(glob.glob(os.path.join(BASE_DIR, '**', 's2_best_model.pth'), recursive=True), key=os.path.getmtime, reverse=True)
        if prev and self.rank == 0: shutil.copy2(prev[0], best_ckpt)

        for epoch in range(self.num_epochs):
            tr_loss, tr_acc = self.train(epoch)
            va_loss, va_acc = self.evaluation(epoch)

            if self.rank == 0:
                summary = (
                    f"S2-DDP Epoch {epoch+1}/{self.num_epochs} SUMMARY | "
                    f"Train Loss: {tr_loss:.4f} (Acc: {tr_acc:.2f}%) | "
                    f"Val Loss: {va_loss:.4f} (Acc: {va_acc:.2f}%)"
                )
                print(summary, flush=True)
                logging.info(summary)

                writer.add_scalar('S2/Train_Loss', tr_loss, epoch)
                writer.add_scalar('S2/Train_Acc', tr_acc, epoch)
                writer.add_scalar('S2/Val_Loss', va_loss, epoch)
                writer.add_scalar('S2/Val_Acc', va_acc, epoch)

                if not math.isnan(va_loss) and va_loss < best_val_loss:
                    best_val_loss = va_loss
                    torch.save(self.net.module.uvit.state_dict(), best_ckpt)

            dist.barrier()

        if self.rank == 0:
            writer.close()

        if os.path.exists(best_ckpt):
            state = torch.load(best_ckpt, map_location='cpu', weights_only=True)
            self.net.module.uvit.load_state_dict(state)

        self.export_stage3_clips()
        if self.rank == 0:
            print("Stage 2 Training & Stage-3 Clip Export 100% Completed!", flush=True)
            logging.info("Stage 2 Training & Stage-3 Clip Export 100% Completed!")


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s: %(message)s')

    s2 = S2Solver(local_rank, world_size)
    s2.run()
    dist.destroy_process_group()
