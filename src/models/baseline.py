import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")

        self.enc0 = nn.Sequential(
            backbone.conv1,  # 64, stride 2
            backbone.bn1,
            backbone.relu,
        )  # 128x128

        self.enc1 = nn.Sequential(
            backbone.maxpool,
            backbone.layer1
        )  # 64x64

        self.enc2 = backbone.layer2  # 32x32
        self.enc3 = backbone.layer3  # 16x16
        self.enc4 = backbone.layer4  # 8x8

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)      # 128
        x1 = self.enc1(x0)     # 64
        x2 = self.enc2(x1)     # 32
        x3 = self.enc3(x2)     # 16
        x4 = self.enc4(x3)     # 8

        d3 = self.dec4(x4, x3)  # 16
        d2 = self.dec3(d3, x2)  # 32
        d1 = self.dec2(d2, x1)  # 64
        d0 = self.dec1(d1, x0)  # 128

        out = self.final(d0)

        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)

        return out
