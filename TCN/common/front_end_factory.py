# TCN/common/front_end_factory.py
import torch.nn as nn

def build_front_end(kind: str,
                    in_channels: int,
                    k: int = 21,
                    stride: int = 1,
                    cutoff_ratio: float = 0.5):
    """
    Minimal factory for your current repo state:
      - 'none'     : identity
      - 'spectral' : SpectralPool1d (frequency truncation; non-causal)
    We keep 'lpsconv' handling in char_cnn_test.py via HartleyTCN,
    because that wrapper composes your front-end with the base TCN.
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "spectral":
        from .frontends.spectral_pool1d import SpectralPool1d
        return SpectralPool1d(cutoff_ratio=float(cutoff_ratio))

    raise ValueError(
        f"Unknown front-end '{kind}'. Available: ['none', 'spectral'].\n"
        f"For 'lpsconv' (your method), use HartleyTCN in the task runner."
    )
