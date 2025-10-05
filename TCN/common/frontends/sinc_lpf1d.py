import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SincLPF1d(nn.Module):
    """
    Learnable low-pass via windowed-sinc kernel (odd length -> linear phase).
    - cutoff in (0, 0.5] as a fraction of Nyquist (pi rad/sample)
    - optional learnable kaiser beta for transition control (sharper stopband)
    """
    def __init__(self, in_ch: int, k: int = 21, cutoff: float = 0.25,
                 learn_cutoff: bool = True, learn_beta: bool = False,
                 causal: bool = False, unity_dc: bool = True):
        super().__init__()
        assert k % 2 == 1, "kernel size must be odd"
        self.in_ch = in_ch
        self.k = k
        self.causal = causal
        self.unity_dc = unity_dc

        # cutoff ∈ (0, 0.5] (Nyquist fraction); parameterize via sigmoid
        c0 = torch.tensor(float(cutoff))
        self._pc = nn.Parameter(torch.logit(torch.clamp(c0, 1e-4, 0.5-1e-4))) if learn_cutoff else None
        self.register_buffer('_cutoff', c0 if not learn_cutoff else torch.tensor(0.0))

        # Kaiser beta
        self._pb = nn.Parameter(torch.tensor(8.0)) if learn_beta else None
        self.register_buffer('_beta', torch.tensor(8.0) if not learn_beta else torch.tensor(0.0))

        # 1x1 mix (optional expressiveness; init identity-like)
        self.mix = nn.Conv1d(in_ch, in_ch, kernel_size=1, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.mix.weight)

    def _kaiser(self, n, beta, N):
        # Approx Kaiser window, n in [0,k-1], centered
        i0 = torch.i0
        num = i0(beta * torch.sqrt(1 - ((2*n/(N-1) - 1)**2)))
        den = i0(torch.tensor(beta))
        return (num / den).to(num.dtype)

    def _make_kernel(self, device, dtype):
        k = self.k
        M = (k - 1) // 2
        n = torch.arange(-M, M+1, device=device, dtype=dtype)

        cutoff = torch.sigmoid(self._pc) if self._pc is not None else self._cutoff
        cutoff = torch.clamp(cutoff, 1e-4, 0.5 - 1e-4)  # safety

        # ideal LPF impulse: 2*f_c * sinc(2*f_c*n)
        fc = cutoff
        h = 2*fc * torch.sinc(2*fc*n)

        beta = self._pb if self._pb is not None else self._beta  # scalar
        # Kaiser window (vector length k)
        t = torch.arange(0, k, device=device, dtype=dtype)
        w = self._kaiser(t, beta, k)
        h = h * w

        if self.unity_dc:
            h = h / (h.sum() + 1e-8)

        # depthwise: same kernel per channel
        h = h.view(1, 1, k).repeat(self.in_ch, 1, 1)  # (C,1,K)
        return h

    def forward(self, x):
        # x: (B,C,T)
        h = self._make_kernel(x.device, x.dtype)
        pad = (self.k - 1)
        if self.causal:
            y = F.conv1d(x, h, bias=None, stride=1, padding=0, groups=self.in_ch)
            y = F.pad(y, (pad, 0))  # left pad only => fixed latency
        else:
            y = F.conv1d(F.pad(x, (pad//2, pad - pad//2), mode='reflect'),
                         h, bias=None, stride=1, padding=0, groups=self.in_ch)
        y = self.mix(y)
        # keep average level (approx)
        if self.unity_dc:
            xm = x.mean(dim=-1, keepdim=True); ym = y.mean(dim=-1, keepdim=True)
            y = y - ym + xm
        return y
