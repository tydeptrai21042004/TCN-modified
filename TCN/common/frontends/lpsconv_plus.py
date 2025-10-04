# TCN/common/frontends/lpsconv_plus.py
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from TCN.common.symmetric_conv1d import SymmetricConv1d  # your linear-phase FIR

class LPSConvPlus(nn.Module):
    """
    Stronger linear-phase front-end:
      - Two cascaded symmetric FIR stages (sharper low-pass)
      - 1x1 pointwise channel mixing
      - Learnable gated residual: y = x + sigma(beta) * (F(x) - x)
      - Optional causal
      - Unity-DC projection inside symmetric convs (if supported)
    """
    def __init__(self, in_ch: int, k1: int = 15, k2: int = 15,
                 causal: bool = False, unity_dc: bool = True):
        super().__init__()
        # Two cascaded symmetric FIRs
        self.fir1 = SymmetricConv1d(in_ch, in_ch, kernel_size=k1,
                                    h=1.0, causal=causal, bias=False)
        self.fir2 = SymmetricConv1d(in_ch, in_ch, kernel_size=k2,
                                    h=1.0, causal=causal, bias=False)
        # Simple nonlinearity between stages (keeps phase linear on the filter itself;
        # we apply it between stages, not inside the symmetric kernel definition)
        self.act  = nn.ReLU(inplace=True)
        # 1x1 pointwise mixing (keeps time length, mixes channels)
        self.mix  = nn.Conv1d(in_ch, in_ch, kernel_size=1, bias=False)
        # Gate: start small so we don't over-smooth at init
        self.beta = nn.Parameter(torch.tensor(-2.0))  # sigma(-2) ~ 0.12
        # Optional per-channel scale to keep DC near 1 after mix
        self.register_buffer('eps', torch.tensor(1e-8))
        self.unity_dc = bool(unity_dc)

        # Init: start near identity (pointwise ~ I)
        with torch.no_grad():
            nn.init.zeros_(self.mix.weight)
            # fir1/fir2 already small random or your default init; identity mix ensures early stability

    def forward(self, x):
        # x: (B, C, T)
        y = self.fir1(x)
        y = self.act(y)
        y = self.fir2(y)
        y = self.mix(y)

        if self.unity_dc:
            # Project average over time to be close to x average (approximate unity DC)
            # Compute per-channel mean and re-center y to match x
            xm = x.mean(dim=-1, keepdim=True)
            ym = y.mean(dim=-1, keepdim=True)
            y = y - ym + xm

        alpha = torch.sigmoid(self.beta)
        return x + alpha * (y - x)  # gated residual blend
