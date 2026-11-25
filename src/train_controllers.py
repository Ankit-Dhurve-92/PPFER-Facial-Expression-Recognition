import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from pytorch_wavelets import DWTForward

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_params
train_params.s1_video_clip_dir = os.path.expanduser("~/datasets/CREMA-D/clips")
train_params.train_identity_file = os.path.expanduser("~/datasets/CREMA-D/train_identity.csv")
train_params.train_emotion_file = os.path.expanduser("~/datasets/CREMA-D/train_emotion.csv")
train_params.test_identity_file = os.path.expanduser("~/datasets/CREMA-D/test_identity.csv")
train_params.test_emotion_file = os.path.expanduser("~/datasets/CREMA-D/test_emotion.csv")

from loader import Cremad

def train_model(task_name, num_classes, subband, save_path, epochs=8):
    print(f"\n=== Training {task_name} ({num_classes} classes, subband={subband}) ===")
    device = torch.device("cuda:0")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = nn.DataParallel(model.to(device), device_ids=[0, 1])

    dwt = DWTForward(J=1, wave='db1').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = Cremad('train', 1)
    val_ds = Cremad('validation', 1)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            B, T, C, H, W = imgs.shape
            # Har clip se 4 evenly spaced frames lete hain fast training ke liye
            imgs_sub = imgs[:, ::4, :, :, :].reshape(-1, C, H, W).to(device)
            if subband == "high":
                _, high = dwt(imgs_sub)
                # 3 high-frequency subbands (LH, HL, HH) ko combine karke 3-channel image banate hain
                x = high[0].mean(dim=2)
                lbl = labels[:, 0].unsqueeze(1).repeat(1, 4).reshape(-1).to(device)
            elif subband == "low":
                low, _ = dwt(imgs_sub)
                x = low
                lbl = labels[:, 1].unsqueeze(1).repeat(1, 4).reshape(-1).to(device)
            else:
                x = imgs_sub
                lbl = labels[:, 1].unsqueeze(1).repeat(1, 4).reshape(-1).to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * lbl.size(0)
            correct += (out.argmax(1) == lbl).sum().item()
            total += lbl.size(0)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                B, T, C, H, W = imgs.shape
                imgs_sub = imgs[:, ::4, :, :, :].reshape(-1, C, H, W).to(device)
                if subband == "high":
                    _, high = dwt(imgs_sub)
                    x = high[0].mean(dim=2)
                    lbl = labels[:, 0].to(device)
                elif subband == "low":
                    low, _ = dwt(imgs_sub)
                    x = low
                    lbl = labels[:, 1].to(device)
                else:
                    x = imgs_sub
                    lbl = labels[:, 1].to(device)

                out = model(x).view(B, 4, -1).mean(dim=1)
                val_correct += (out.argmax(1) == lbl).sum().item()
                val_total += B

        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Clip Acc: {val_acc:.2f}%")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.module.state_dict(), save_path)
    print(f"Saved best {task_name} (Val Acc: {best_acc:.2f}%) -> {save_path}")

if __name__ == "__main__":
    os.makedirs("pretrained_weights", exist_ok=True)
    train_model("c_hpr (Identity Controller)", 91, "high", "pretrained_weights/c_hpr_identity.pth", epochs=8)
    train_model("c_lpr (Emotion Controller)", 6, "low", "pretrained_weights/c_lpr_emotion.pth", epochs=8)
    train_model("Stage-2 FER Controller", 6, "rgb", "pretrained_weights/cremad_fer_controller.pth", epochs=8)
