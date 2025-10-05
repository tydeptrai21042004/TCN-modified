# TCN/common/frontends/fir_remez.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _hann(k, device=None, dtype=None):
    n = torch.arange(k, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (k - 1))

def _kaiser_beta(att_db: float):
    # Kaiser beta approximation
    a = abs(att_db)
    if a > 50: return 0.1102 * (a - 8.7)
    if a >= 21: return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    return 0.0

def _kaiser_window(k: int, beta: float, device=None, dtype=None):
    n = torch.arange(0, k, device=device, dtype=dtype)
    return torch.i0(beta * torch.sqrt(1 - ((2*n)/(k-1) - 1) ** 2)) / torch.i0(torch.tensor(beta))

def _sinc(x):
    return torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))

class FIRRemezLPF1d(nn.Module):
    """
    Fixed equiripple (Parks–McClellan) low-pass FIR. If SciPy is unavailable,
    falls back to windowed-sinc with Kaiser window.
    cutoff_ratio: normalized cutoff in (0, 1) where 1.0 ~ Nyquist (0.5 cycles/sample).
    """
    def __init__(self, in_ch: int, k: int = 63, cutoff_ratio: float = 0.5,
                 att_db: float = 60.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.in_ch = int(in_ch)
        self.k = int(k)
        self.cut = float(cutoff_ratio)
        self.att_db = float(att_db)

        weight = self._design_kernel()
        # register as buffer, depthwise weights (in_ch groups)
        w = weight.view(1, 1, -1).repeat(self.in_ch, 1, 1)  # (in_ch,1,k)
        self.register_buffer("weight", w, persistent=False)

    def _design_kernel(self) -> torch.Tensor:
        try:
            from scipy.signal import remez
            # remez expects bands in [0, 0.5] for normalized freq with Hz=1
            bands = [0.0, float(self.cut) * 0.5, float(self.cut) * 0.5 + 0.05, 0.5]
            desired = [1.0, 0.0]
            taps = remez(self.k, bands=bands, desired=desired, Hz=1.0, maxiter=100)
            h = torch.from_numpy(taps).float()
        except Exception:
            # Fallback: windowed-sinc (Kaiser)
            n = torch.arange(-(self.k // 2), (self.k // 2) + 1).float()
            fc = 0.5 * float(self.cut)  # map (0..1) -> (0..0.5)
            h = 2.0 * fc * _sinc(2.0 * fc * n)
            beta = _kaiser_beta(self.att_db)
            w = _kaiser_window(self.k, beta, dtype=torch.float32)
            h = h * w
        # normalize DC gain to 1
        h = h / (h.sum() + 1e-8)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        pad = self.k // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        return F.conv1d(x, self.weight, bias=None, stride=1, padding=0, groups=self.in_ch)
