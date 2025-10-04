import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricConv1d(nn.Module):
    r"""
    Symmetric (even) 1-D convolution implementing the h-Hartley-cosine rule:

        (f * g)(n h) = (h/2) ∑_m f(m h) [ g(n h + m h) + g(n h - m h) ]

    For an even kernel f[m] = f[-m], this equals h * (standard discrete convolution of f with g).

    Args:
        in_channels  (int): number of input channels C_in
        out_channels (int): number of output channels C_out
        kernel_size  (int): odd kernel length K = 2k+1
        h           (float): scalar factor in the definition above
        bias        (bool): include bias per out_channel
        causal      (bool): if True, left-pad so output at t uses x[≤ t] only
        dilation    (int): dilation for the filter
        groups      (int): conv groups (default 1). Must divide in_channels and out_channels.

    Input shape:  (N, C_in, L)  •  Output shape: (N, C_out, L)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 h: float = 1.0,
                 bias: bool = False,
                 causal: bool = False,
                 dilation: int = 1,
                 groups: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd (Type-I linear-phase)"
        assert in_channels % groups == 0 and out_channels % groups == 0, "groups must divide channels"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.ks           = kernel_size
        self.h            = float(h)
        self.causal       = bool(causal)
        self.dilation     = int(dilation)
        self.groups       = int(groups)

        half = kernel_size // 2  # number of taps on one side (excluding center)
        # Parameterize only half + center: shape = (C_out, C_in/groups, half+1)
        self.w_half = nn.Parameter(torch.empty(out_channels, in_channels // groups, half + 1))
        self.bias   = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # He/Kaiming normal init for learnable taps (like Keras HeNormal, noting TF uses truncated normal) 
        nn.init.kaiming_normal_(self.w_half, mode="fan_in", nonlinearity="relu")  # gain≈√2
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _build_full_kernel(self) -> torch.Tensor:
        # Mirror taps to enforce even symmetry across the kernel axis
        # w_half: [C_out, C_in/groups, half+1]  ->  full: [C_out, C_in/groups, K]
        w_left  = self.w_half[..., 1:].flip(-1)            # exclude center, reverse
        w_full  = torch.cat([w_left, self.w_half], dim=-1) # [C_out, C_in/groups, K]
        return w_full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C_in, L]
        """
        W = self._build_full_kernel()                      # [C_out, C_in/groups, K]
        if self.causal:
            # left pad only so output at t sees x[≤ t]
            pad_left  = self.dilation * (self.ks - 1)
            x_padded  = F.pad(x, (pad_left, 0))
            y = F.conv1d(x_padded, W, bias=self.bias, stride=1,
                         padding=0, dilation=self.dilation, groups=self.groups)
        else:
            # symmetric pad (“same” length when stride=1, odd K) 
            # PyTorch Conv1d docs: padding preserves length for stride=1 with appropriate value.
            # Here we compute it directly to avoid relying on string 'same'. 
            pad = (self.dilation * (self.ks - 1)) // 2
            y = F.conv1d(x, W, bias=self.bias, stride=1,
                         padding=pad, dilation=self.dilation, groups=self.groups)

        # Exactly match the Hartley–cosine definition (even kernel ⇒ correlation == convolution)
        return y * self.h
