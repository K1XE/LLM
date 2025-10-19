from torch import nn, Tensor
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mid_dim = hidden_dim * 8 // 3
        self.gate = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.up = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.down = nn.Linear(mid_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X: Tensor):
        out = self.dropout(self.down(F.silu(self.gate(X)) * self.up(X)))
        return out
