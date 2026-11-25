import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_r3d
from loader import Cremad
from train_params import (
    log_path, tensorboard_log_path, output_path_time,
    num_epochs, learning_rate, gpu_ids
)


class S3Solver:
    def __init__(self):
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.batch_size = 16

        os.makedirs(output_path_time, exist_ok=True)
        logging.basicConfig(
            filename=log_path, level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.model = get_r3d().to('cuda')
        if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.scaler = torch.amp.GradScaler('cuda')

        self.train_ds = Cremad('train', 3)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True
        )
        self.val_ds = Cremad('validation', 3)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True
        )
        print(f"Stage 3 Ready | Train clips: {len(self.train_ds)} | Val clips: {len(self.val_ds)}", flush=True)

    def train(self, epoch: int, writer: SummaryWriter):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for i, (images, labels) in enumerate(self.train_loader):
            # (B, T, C, H, W) -> (B, C, T, H, W)
            videos = images.permute(0, 2, 1, 3, 4).contiguous().to('cuda', non_blocking=True)
            targets = labels[:, 1].to('cuda', non_blocking=True)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = self.model(videos)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

            if i % 100 == 0:
                print(
                    f"S3 Epoch {epoch+1}/{self.num_epochs} | Step {i}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | Running Acc: {100.0 * correct / total:.2f}%",
                    flush=True
                )

        avg_loss = total_loss / total
        avg_acc = 100.0 * correct / total
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('train_accuracy', avg_acc, epoch)
        return avg_loss, avg_acc

    def validate(self, epoch: int, writer: SummaryWriter):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                videos = images.permute(0, 2, 1, 3, 4).contiguous().to('cuda', non_blocking=True)
                targets = labels[:, 1].to('cuda', non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = self.model(videos)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item() * targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / total
        avg_acc = 100.0 * correct / total
        writer.add_scalar('val_loss', avg_loss, epoch)
        writer.add_scalar('val_accuracy', avg_acc, epoch)
        return avg_loss, avg_acc

    def run(self):
        best_val_acc = 0.0
        writer = SummaryWriter(tensorboard_log_path)
        for epoch in range(self.num_epochs):
            tr_loss, tr_acc = self.train(epoch, writer)
            va_loss, va_acc = self.validate(epoch, writer)
            self.scheduler.step()

            msg = (
                f"S3 Epoch {epoch+1}/{self.num_epochs} SUMMARY | "
                f"Train Loss: {tr_loss:.4f} (Acc: {tr_acc:.2f}%) | "
                f"Val Loss: {va_loss:.4f} (Acc: {va_acc:.2f}%)"
            )
            print(msg, flush=True)
            logging.info(msg)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(state, os.path.join(output_path_time, 's3_best_model.pth'))
                print(f"--> New Best S3 Model Saved! Val Acc: {best_val_acc:.2f}%", flush=True)

        print(f"\nStage 3 Training 100% Completed! Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
        print(f"Saved at: {output_path_time}/s3_best_model.pth", flush=True)


if __name__ == '__main__':
    S3Solver().run()
