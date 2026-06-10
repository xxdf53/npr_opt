import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learnable spatial attention pooling.

    Instead of uniformly averaging all spatial positions (GAP), this module
    learns a per-position importance weight via a lightweight 1x1 conv, then
    computes a weighted average. The attention map can be extracted and
    interpolated to the original image size as a forgery heatmap.
    """

    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        score = self.sigmoid(self.attn(x))       # [B, 1, H, W]
        self.attention_map = score                # stored for heatmap extraction
        numerator = (x * score).sum(dim=[2, 3], keepdim=True)   # [B, C, 1, 1]
        denominator = score.sum(dim=[2, 3], keepdim=True) + 1e-8
        return numerator / denominator           # [B, C, 1, 1]
