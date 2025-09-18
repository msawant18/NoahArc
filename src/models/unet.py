"""
UNet model for segmentation.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        # encoder (simple conv stack, not pretrained to keep dependencies minimal)
        self.enc1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.up4 = UpBlock(features[3]*2, features[3])
        self.up3 = UpBlock(features[3], features[2])
        self.up2 = UpBlock(features[2], features[1])
        self.up1 = UpBlock(features[1], features[0])

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.head(d1)
        return out  # logits
