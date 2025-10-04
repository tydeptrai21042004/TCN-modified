# frontends/spectral_pool1d.py
import torch
import torch.nn as nn
import torch.fft as fft

class SpectralPool1d(nn.Module):
    """
    Frequency-domain truncation (ideal LPF).
    - cutoff_ratio in (0,1]: keep that fraction of Nyquist band.
    - Non-causal (uses global FFT); use only in non-causal pipelines.
    """
    def __init__(self, cutoff_ratio: float = 0.5):
        super().__init__()
        assert 0.0 < cutoff_ratio <= 1.0
        self.cutoff_ratio = cutoff_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        B, C, T = x.shape
        X = fft.rfft(x, dim=-1)  # (B,C,F) where F=T//2+1
        F = X.shape[-1]
        keep = int(max(1, round(self.cutoff_ratio * F)))
        # Zero out high freqs
        X[..., keep:] = 0
        y = fft.irfft(X, n=T, dim=-1)
        return y.to(x.dtype)
