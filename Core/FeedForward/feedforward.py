# Core/FeedForward/feedforward.py — Naylis v1
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Feed-Forward Network avec SwiGLU.
    hidden_dim = (8/3 * embed_dim) arrondi au multiple de 64.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0, use_swiglu: bool = True):
        super().__init__()
        self.embed_dim  = embed_dim
        self.use_swiglu = use_swiglu

        if use_swiglu:
            self.hidden_dim = (int(8 * embed_dim / 3) + 63) // 64 * 64
            self.gate_proj  = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            self.up_proj    = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            self.down_proj  = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        else:
            self.hidden_dim = 4 * embed_dim
            self.fc1        = nn.Linear(embed_dim, self.hidden_dim)
            self.fc2        = nn.Linear(self.hidden_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)
        return self.dropout(x)
