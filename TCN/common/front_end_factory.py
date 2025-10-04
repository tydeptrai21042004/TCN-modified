# TCN/common/front_end_factory.py
import torch.nn as nn

def _odd(n: int) -> int:
    """Make n odd by adding 1 if it is even; also guard small sizes."""
    n = max(3, int(n))
    return n if (n % 2 == 1) else n + 1

def build_front_end(kind: str,
                    in_channels: int,
                    k: int = 21,
                    stride: int = 1,
                    cutoff_ratio: float = 0.5):
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "spectral":
        from .frontends.spectral_pool1d import SpectralPool1d
        return SpectralPool1d(cutoff_ratio=float(cutoff_ratio))
    if kind == "lpsconv_plus":
        from .frontends.lpsconv_plus import LPSConvPlus
        # derive two shorter, ALWAYS-ODD kernels from k
        k_half = _odd(max(7, k // 2))   # e.g., k=21 -> k_half=11 (odd)
        return LPSConvPlus(in_ch=in_channels, k1=k_half, k2=k_half,
                           causal=False, unity_dc=True)
    raise ValueError(f"Unknown front-end: {kind}")
