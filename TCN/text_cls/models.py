# TCN/text_cls/models.py
import torch
import torch.nn as nn

try:
    # official implementation ships a TemporalConvNet in TCN/tcn.py
    from TCN.tcn import TemporalConvNet
except Exception:
    # fallback: nested import if your local path differs
    from tcn import TemporalConvNet  # pragma: no cover

class TextTCN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 channels=(128, 128, 128), kernel_size=3,
                 dropout=0.1, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.tcn = TemporalConvNet(
            num_inputs=embed_dim,
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, tokens):            # tokens: (B, L)
        x = self.embedding(tokens)        # (B, L, E)
        x = x.transpose(1, 2)             # (B, E, L) for Conv1d/TCN
        y = self.tcn(x)                   # (B, C, L)
        y_last = y[:, :, -1]              # causal TCN → use last timestep
        return self.classifier(y_last)
