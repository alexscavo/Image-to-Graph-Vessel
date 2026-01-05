import torch.nn as nn
import torch.nn.functional as F

class SegHead2D(nn.Module):
    def __init__(self, in_ch, mid_ch=64, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1)  # logits
        )

    def forward(self, x, out_size):
        x = self.conv(x)   # [B,out_ch,H',W']
        x = F.interpolate(
            x, size=out_size,
            mode="bilinear", align_corners=False
        )                  # [B,out_ch,H,W]
        return x
