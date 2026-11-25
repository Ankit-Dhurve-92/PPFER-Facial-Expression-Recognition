import os
import sys
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from train_params import (
    unet_pretrain_weight, resnet50_pretrain_weight, c_lpr_pretrain_weight,
    best_f_hpr, best_f_lpr, image_size, dfew_image_fer_weight, uvit_256_weight
)

uvit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'U-ViT'))
if uvit_path not in sys.path:
    sys.path.insert(0, uvit_path)

from libs.uvit import UViT


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def _load_clean_state(model, ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)


def get_unet():
    model = UNet(n_channels=3, n_classes=3)
    if os.path.exists(unet_pretrain_weight):
        _load_clean_state(model, unet_pretrain_weight)
    return model


def get_controller(num_classes=91, ckpt_path=None):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    target_path = ckpt_path if ckpt_path else resnet50_pretrain_weight
    if os.path.exists(target_path):
        _load_clean_state(model, target_path)
    return model


def get_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 6)
    return model


def get_r3d():
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 6)
    return model


def get_best_f_hpr():
    model = UNet(n_channels=3, n_classes=3)
    _load_clean_state(model, best_f_hpr)
    return model


def get_best_f_lpr():
    model = UNet(n_channels=3, n_classes=3)
    _load_clean_state(model, best_f_lpr)
    return model


def get_uvit():
    img_size = image_size // 8
    nnet = UViT(
        img_size=img_size, patch_size=2, in_chans=4,
        embed_dim=1152, depth=28, num_heads=16, num_classes=1001, conv=False
    )
    if os.path.exists(uvit_256_weight):
        _load_clean_state(nnet, uvit_256_weight)
    return nnet


def get_feature_controller():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    if os.path.exists(dfew_image_fer_weight):
        _load_clean_state(model, dfew_image_fer_weight)
    return model
