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
                    cutoff_ratio: float = 0.5,
                    bands: int = 8):
    """
    kind: 'none' | 'spectral' | 'lpsconv_plus' | 'lpsconv_sinc' | 'sincnet_bank' | 'fir_remez' | 'blurpool'
    k: kernel length (odd enforced where applicable)
    stride: used by blurpool; otherwise 1
    cutoff_ratio: normalized cutoff in (0, 1), used by spectral & fir_remez (as low-pass)
    bands: number of band-pass filters (for sincnet_bank)
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()

    if kind == "spectral":
        from .frontends.spectral_pool1d import SpectralPool1d
        return SpectralPool1d(cutoff_ratio=float(cutoff_ratio))

    if kind == "lpsconv_plus":
        from .frontends.lpsconv_plus import LPSConvPlus
        k_half = _odd(max(7, k // 2))
        return LPSConvPlus(in_ch=in_channels, k1=k_half, k2=k_half,
                           causal=True, unity_dc=False)

    if kind == "lpsconv_sinc":
        from .frontends.sinc_lpf1d import SincLPF1d
        k = _odd(k)
        return SincLPF1d(in_ch=in_channels, k=k, cutoff=0.25,
                         learn_cutoff=True, learn_beta=True,
                         causal=True, unity_dc=False)

    if kind == "sincnet_bank":
        from .frontends.sincnet_bank import SincNetBank1d
        k = _odd(k)
        return SincNetBank1d(in_ch=in_channels, bands=int(bands), k=k, causal=False)

    if kind == "fir_remez":
        from .frontends.fir_remez import FIRRemezLPF1d
        k = _odd(k)
        return FIRRemezLPF1d(in_ch=in_channels, k=k, cutoff_ratio=float(cutoff_ratio))

    if kind == "blurpool":
        from .frontends.blurpool1d import BlurPool1d
        k = _odd(k)
        return BlurPool1d(in_ch=in_channels, k=k, stride=max(1, int(stride)))

    raise ValueError(f"Unknown front-end: {kind}")
