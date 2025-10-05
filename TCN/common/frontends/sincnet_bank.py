# TCN/common/frontends/sincnet_bank.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _hann_window(k: int, device=None, dtype=None):
    n = torch.arange(k, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (k - 1))

def _sinc(x):
    # normalized sinc: sin(pi x)/(pi x); define at 0 by limit 1
    y = torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    return y

class SincNetBank1d(nn.Module):
    """
    Depthwise Sinc band-pass filterbank:
      - Learn f_low and f_high per band (normalized in (0, 0.5))
      - Hann-windowed ideal BPF: h_bp = 2*f2*sinc(2*f2*t) - 2*f1*sinc(2*f1*t)
      - Outputs in_ch * bands channels, then mixes back to in_ch via 1x1 conv.
    """
    def __init__(self, in_ch: int, bands: int = 8, k: int = 63,
                 min_low: float = 0.0, min_bw: float = 0.01,
                 causal: bool = False):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.in_ch = in_ch
        self.bands = int(bands)
        self.k = int(k)
        self.min_low = float(min_low)
        self.min_bw = float(min_bw)
        self.causal = bool(causal)

        # Learnable frequencies in normalized [0, 0.5): parameterize via sigmoid
        # Initialize with roughly mel-spaced bands in (0, 0.5)
        f1_init = torch.linspace(0.05, 0.40, steps=self.bands)
        bw_init = torch.full((self.bands,), 0.05)
        # store logits
        self.f1_ = nn.Parameter(torch.logit(torch.clamp(f1_init, 1e-4, 0.4999)))
        self.bw_ = nn.Parameter(torch.logit(torch.clamp(bw_init, 1e-4, 0.4999)))
        # 1x1 channel mix to map (in_ch*bands) -> in_ch
        self.mix = nn.Conv1d(in_ch * self.bands, in_ch, kernel_size=1, groups=1, bias=True)

        # register fixed buffers for time grid and window
        t = torch.arange(-(k // 2), (k // 2) + 1).float()
        self.register_buffer("t_grid", t, persistent=False)
        self.register_buffer("window", _hann_window(k, dtype=torch.float32), persistent=False)

    def _freqs(self):
        # constrain to valid region and ensure f_high > f_low + min_bw
        f1 = torch.sigmoid(self.f1_) * 0.5   # (0, 0.5)
        bw = torch.sigmoid(self.bw_) * 0.5
        f1 = torch.clamp(f1, self.min_low, 0.5 - self.min_bw - 1e-4)
        f2 = torch.clamp(f1 + torch.clamp(bw, self.min_bw, 0.5 - f1 - 1e-4),
                         self.min_low + self.min_bw, 0.4999)
        return f1, f2  # shape: (bands,)

    def _make_kernels(self, device, dtype):
        t = self.t_grid.to(device=device, dtype=dtype)
        w = self.window.to(device=device, dtype=dtype)
        f1, f2 = self._freqs()  # (bands,)
        # each lp: h_lp(fc) = 2*fc * sinc(2*fc*t)
        # use broadcasting: (bands, k)
        t_b = t.unsqueeze(0).expand(self.bands, -1)
        h1 = (2.0 * f1.unsqueeze(1)) * _sinc(2.0 * f1.unsqueeze(1) * t_b)
        h2 = (2.0 * f2.unsqueeze(1)) * _sinc(2.0 * f2.unsqueeze(1) * t_b)
        hbp = (h2 - h1) * w.unsqueeze(0)  # (bands, k)
        # normalize each band to unit L1 (DC-neutrality for band-pass)
        hbp = hbp - hbp.mean(dim=1, keepdim=True)  # zero-mean
        # shape for depthwise grouped conv: (out_ch, 1, k)
        # out_ch = in_ch * bands
        hbp = hbp.unsqueeze(1).repeat(self.in_ch, 1, 1)  # (in_ch*bands, 1, k)
        return hbp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=in_ch, T)
        B, C, T = x.shape
        device, dtype = x.device, x.dtype
        weight = self._make_kernels(device, dtype)
        pad_left = self.k - 1 if self.causal else self.k // 2
        pad_right = 0 if self.causal else self.k // 2
        x_pad = F.pad(x, (pad_left, pad_right), mode="reflect")
        y = F.conv1d(x_pad, weight, bias=None, stride=1, padding=0,
                     groups=self.in_ch)  # (B, in_ch*bands, T)
        y = self.mix(y)  # (B, in_ch, T)
        return y
