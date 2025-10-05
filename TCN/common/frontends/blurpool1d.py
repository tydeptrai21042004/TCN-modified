# TCN/common/frontends/blurpool1d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import comb

def _binomial_row(k: int):
    row = torch.tensor([comb(k - 1, i) for i in range(k)], dtype=torch.float32)
    row = row / row.sum()
    return row

class BlurPool1d(nn.Module):
    """
    Fixed binomial low-pass before (optional) subsampling.
    stride=1 -> just blur; stride>1 -> blur + downsample (anti-aliased).
    """
    def __init__(self, in_ch: int, k: int = 5, stride: int = 1):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.in_ch = int(in_ch)
        self.k = int(k)
        self.stride = int(stride)
        kernel = _binomial_row(self.k).view(1, 1, -1)  # (1,1,k)
        weight = kernel.repeat(in_ch, 1, 1)            # (C,1,K)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.k // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        return F.conv1d(x, self.weight, bias=None, stride=self.stride, padding=0, groups=self.in_ch)
