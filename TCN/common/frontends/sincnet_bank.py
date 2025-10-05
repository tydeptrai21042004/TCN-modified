# TCN/common/frontends/sincnet_bank.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- utils ----------------

def _hann_window(k: int, device=None, dtype=None):
    n = torch.arange(k, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (k - 1))

def _sinc(x: torch.Tensor) -> torch.Tensor:
    # normalized sinc: sin(pi x)/(pi x); define at 0 by limit 1
    return torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))


# ---------------- module ----------------

class SincNetBank1d(nn.Module):
    """
    Depthwise Sinc band-pass filterbank (time-domain, linear-phase when k is odd).
      - Learn f_low (f1) & bandwidth (bw) per band; f_high = f1 + bw
      - All frequencies normalized to Nyquist in (0, 0.5)
      - Each band kernel: h_bp = 2*f2*sinc(2*f2*t) - 2*f1*sinc(2*f1*t), Hann-windowed
      - Applies grouped depthwise conv to produce (in_ch * bands) channels,
        then a 1x1 'mix' conv maps back to in_ch.

    Args
    ----
    in_ch:   input channels
    bands:   number of band-pass filters per input channel
    k:       odd kernel length (linear phase)
    min_low: minimum f1 (lower edge)
    min_bw:  minimum bandwidth (enforced per band)
    causal:  causal padding if True; else symmetric 'reflect' padding (offline)
    """
    def __init__(self,
                 in_ch: int,
                 bands: int = 8,
                 k: int = 63,
                 min_low: float = 0.0,
                 min_bw: float = 0.01,
                 causal: bool = False):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd (Type-I linear phase)"
        self.in_ch = int(in_ch)
        self.bands = int(bands)
        self.k = int(k)
        self.min_low = float(min_low)
        self.min_bw = float(min_bw)
        self.causal = bool(causal)

        # Learnable frequencies (logits); init roughly spaced in (0, 0.5)
        f1_init = torch.linspace(0.05, 0.40, steps=self.bands)  # lower edges
        bw_init = torch.full((self.bands,), 0.05)               # bandwidths

        # store logits (inverse-sigmoid)
        self.f1_ = nn.Parameter(torch.logit(torch.clamp(f1_init, 1e-4, 0.4999)))
        self.bw_ = nn.Parameter(torch.logit(torch.clamp(bw_init, 1e-4, 0.4999)))

        # 1x1 channel mix to map (in_ch * bands) -> in_ch
        self.mix = nn.Conv1d(self.in_ch * self.bands, self.in_ch, kernel_size=1, groups=1, bias=True)
        with torch.no_grad():
            # start near identity contribution from each input channel
            nn.init.zeros_(self.mix.weight)

        # fixed buffers for time grid & window (dtype set at runtime)
        t = torch.arange(-(k // 2), (k // 2) + 1).float()  # [-M, ..., M]
        self.register_buffer("t_grid", t, persistent=False)
        self.register_buffer("window", _hann_window(k, dtype=torch.float32), persistent=False)

    # ---- frequency parameterization ----
    def _freqs(self):
        """
        Returns f1, f2 tensors with shape (bands,), both in (0, 0.5).
        Enforces:
          - f1 >= min_low
          - bw >= min_bw
          - f2 = f1 + bw <= 0.5 - 1e-4
        """
        # base unconstrained in (0, 0.5)
        f1 = torch.sigmoid(self.f1_) * 0.5       # (bands,)
        bw = torch.sigmoid(self.bw_) * 0.5       # (bands,)

        # clamp f1 to [min_low, 0.5 - min_bw - eps]
        eps = 1e-4
        f1 = torch.clamp(f1, min=self.min_low, max=0.5 - self.min_bw - eps)

        # --- IMPORTANT FIX: make clamp bounds tensors (same device/dtype) ---
        # bw ∈ [min_bw, 0.5 - f1 - eps]
        min_bw_t = torch.as_tensor(self.min_bw, device=f1.device, dtype=f1.dtype).expand_as(bw)
        max_bw_t = (0.5 - f1 - eps)
        # ensure non-degenerate upper bound
        max_bw_t = torch.clamp(max_bw_t, min=1e-6)
        # also ensure max ≥ min + tiny eps component-wise
        max_bw_t = torch.maximum(max_bw_t, min_bw_t + 1e-6)

        bw = torch.clamp(bw, min=min_bw_t, max=max_bw_t)

        f2 = f1 + bw
        f2 = torch.clamp(f2, min=self.min_low + self.min_bw, max=0.5 - eps)
        return f1, f2  # (bands,)

    # ---- kernel builder ----
    def _make_kernels(self, device, dtype):
        """
        Build band-pass kernels for grouped depthwise conv.
        Returns weight with shape (in_ch * bands, 1, k).
        """
        t = self.t_grid.to(device=device, dtype=dtype)
        w = self.window.to(device=device, dtype=dtype)
        f1, f2 = self._freqs()  # (bands,)

        # h_lp(fc) = 2*fc * sinc(2*fc*t)
        t_b = t.unsqueeze(0).expand(self.bands, -1)       # (bands, k)
        h1 = (2.0 * f1.unsqueeze(1)) * _sinc(2.0 * f1.unsqueeze(1) * t_b)  # (bands, k)
        h2 = (2.0 * f2.unsqueeze(1)) * _sinc(2.0 * f2.unsqueeze(1) * t_b)  # (bands, k)
        hbp = (h2 - h1) * w.unsqueeze(0)                  # (bands, k)

        # band-pass: zero-mean per band (DC-neutral)
        hbp = hbp - hbp.mean(dim=1, keepdim=True)

        # shape for grouped depthwise conv: out_ch=in_ch*bands, in_ch/group=1
        # replicate the bands for each input channel group
        hbp = hbp.unsqueeze(1).repeat(self.in_ch, 1, 1)   # (in_ch*bands, 1, k)
        return hbp

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=in_ch, T) → y: (B, C=in_ch, T)
        """
        B, C, T = x.shape
        device, dtype = x.device, x.dtype

        weight = self._make_kernels(device, dtype)  # (in_ch*bands, 1, k)

        # padding (causal or symmetric)
        if self.causal:
            pad_left, pad_right = self.k - 1, 0
        else:
            pad_left = pad_right = self.k // 2

        x_pad = F.pad(x, (pad_left, pad_right), mode="reflect")

        # depthwise conv: groups = in_ch; produces in_ch*bands channels
        y = F.conv1d(x_pad, weight, bias=None, stride=1, padding=0, groups=self.in_ch)  # (B, in_ch*bands, T)

        # mix back to in_ch
        y = self.mix(y)  # (B, in_ch, T)
        return y
