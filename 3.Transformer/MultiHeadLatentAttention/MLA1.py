import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_len=128, base=10_000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.max_len = max_len
        self.base = base
        self.sin, self.cos = self._compute_pos_emb()

    def _compute_pos_emb(self):
        from math import log

        pos = torch.arange(0, self.max_len)
        div_term = torch.exp(
            torch.arange(0, self.head_dim, 2) * (log(self.base) / self.head_dim)
        )
        theta = torch.outer(pos, div_term)
        sin, cos = theta.sin(), theta.sin()

        return sin, cos

    def forward(self, X: Tensor):
        bsz, num_heads, seq_len, head_dim = X.shape
        self.cos = self.cos[None, None, :seq_len, self.head_dim]
        self.sin = self.sin[None, None, :seq_len, self.head_dim]
        X_ = X.reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
        X1 = X_[..., 0]
        X2 = X_[..., 1]
        out1 = X1 * self.cos - X2 * self.sin
        out2 = X2 * self.cos + X1 * self.cos
        output = torch.stack([out1, out2], dim=-1)
        return output


class MLA(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        down_dim=128,
        up_dim=256,
        rope_dim=56,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.v_head_dim = up_dim // num_heads
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.rope_dim = rope_dim
        self.dropout_rate = dropout_rate
        self.down_proj_kv = nn.Linear(hidden_dim, down_dim)
        self.down_proj_q = nn.Linear(hidden_dim, down_dim)
        self.up_proj_k = nn.Linear(down_dim, up_dim)
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)
        self.proj_qr = nn.Linear(down_dim, rope_dim * num_heads)
        self.proj_kr = nn.Linear(hidden_dim, rope_dim)
        self.rope_q = RotaryEmbedding(rope_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_heads * self.v_head_dim, hidden_dim)
        self.res_dropout = nn.Dropout(dropout_rate)

    def forward(self, X: Tensor, mask=None):
        bsz, seq_len, _ = X.shape
        c_kv = self.down_proj_kv(X)
        k_c = self.up_proj_k(c_kv)
        v_c = self.up_proj_v(c_kv)
        c_q = self.down_proj_q(X)
        q_c = self.up_proj_q(c_q)

        q_r = self.proj_qr(self.rope_q(c_q))
        k_r = self.proj_kr(self.rope_k(X))

        q_c = q_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        q = torch.cat([q_c, q_r], dim=-1)
        k_c = k_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        k_r = k_r.repeat(1, self.num_heads, 1, 1)
        k = torch.cat([k_c, k_r], dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (
            self.head_dim**0.5 + self.rope_dim**0.5
        )
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v_c = v_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        output = (
            torch.matmul(attn_weights, v_c).transpose(1, 2).reshape(bsz, seq_len, -1)
        )
        output = self.res_dropout(self.fc(output))

        return output
