import torch
from torch import nn, Tensor


class RotaryEmbedding(nn.Module):
    def __init__(
        self, hidden_dim, num_heads, max_len=512, base=10_000, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim / num_heads
        self.base = base
        self.max_len = max_len
        self.sin, self.cos = self._compute_pos_emb()

    def _compute_pos_emb(self):
        from math import log

        pos = torch.arange(0, self.max_len)
        div_term = torch.exp(
            torch.arange(0, self.head_dim, 2) * (-log(self.base) / self.head_dim)
        )
        theta = torch.outer(pos, div_term)
        sin = theta.sin().repeat_interleave(2, dim=-1)
        cos = theta.cos().repeat_interleave(2, dim=-1)
        return sin, cos

    def forward(self, q: Tensor):
        bsz, num_heads, seq_len, head_dim = q.shape
        sin, cos = self.sin[:seq_len].unsqueeze(0).unsqueeze(0), self.cos[
            :seq_len
        ].unsqueeze(0).unsqueeze(0)
        q1 = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1)
        output = q * cos + q1 * sin
        return output

"""
class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_len, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_len = max_len
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim / num_heads
        self.sin, self.cos = self._compute_pos_emb()

    def _compute_pos_emb(self):
        from math import log

        theta = torch.arange(0, self.max_len)
        div_term = torch.exp(
            torch.arange(0, self.head_dim, 2) * (-log(10_000) / self.head_dim)
        )
        pos_emb = torch.outer(theta, div_term)
        sin, cos = pos_emb.sin(), pos_emb.cos()
        return sin, cos


    def forward(self, X: Tensor):
        bsz, num_heads, seq_len, head_dim = X.shape
        sin, cos = self.sin[None, None, :seq_len, :], self.cos[:seq_len].unsqueeze(
            0
        ).unsqueeze(0)
        X_ = X.reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
        X1 = X_[..., 0]
        X2 = X_[..., 1]
        o1 = X1 * cos - X2 * sin
        o2 = X2 * cos + X1 * cos
        o = torch.stack([o1, o2], dim=-1)
        out = o.reshape(bsz, num_heads, seq_len, head_dim)
        return out
"""
