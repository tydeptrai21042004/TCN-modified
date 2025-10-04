# TCN/common/hartley_tcn.py
import torch.nn as nn
from TCN.common.symmetric_conv1d import SymmetricConv1d

class HartleyTCN(nn.Module):
    """
    Wrap a TCN; optionally apply a symmetric front-end before it.
    If residual=True and in==out channels, uses x + front(x).
    """
    def __init__(self, base_tcn, in_channels,
                 use_front=False, k=9, h=1.0, causal=False, residual=True):
        super().__init__()
        self.base = base_tcn
        self.use_front = bool(use_front)
        self.residual  = bool(residual)
        if self.use_front:
            # Keep channels the same so residual add is valid for Adding problem (C_in=2)
            self.front = SymmetricConv1d(in_channels, in_channels, kernel_size=k,
                                         h=h, causal=causal, bias=False)

    def forward(self, x):
        if self.use_front:
            y = self.front(x)
            x = (x + y) if self.residual else y
        return self.base(x)
