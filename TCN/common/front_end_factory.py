# TCN/common/front_end_factory.py
import torch.nn as nn

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
        # default: two K=15 stages; feel free to expose CLI args if you want
        return LPSConvPlus(in_ch=in_channels, k1=max(7, k//2), k2=max(7, k//2),
                           causal=False, unity_dc=True)
    raise ValueError(f"Unknown front-end: {kind}")
