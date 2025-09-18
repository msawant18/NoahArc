"""
UNet model for segmentation.
"""
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # TODO: implement full UNet
        self.dummy = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return torch.sigmoid(self.dummy(x))
