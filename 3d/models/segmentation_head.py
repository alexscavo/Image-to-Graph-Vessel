import torch.nn as nn
import torch.nn.functional as F

class SegHead3D(nn.Module):
    def __init__(self, in_ch, mid_ch=64, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, 1)  # logits
        )

    def forward(self, x, out_size):  # out_size: (D,H,W)
        x = self.conv(x)  # [B,out_ch,D',H',W']
        x = F.interpolate(
            x, size=out_size,
            mode="trilinear", align_corners=False
        )  # [B,out_ch,D,H,W]
        return x
