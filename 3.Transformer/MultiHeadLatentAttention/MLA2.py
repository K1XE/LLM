# https://zhuanlan.zhihu.com/p/715155329
import torch

from torch import nn, Tensor

class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_len=1024, base=10_000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_len = max_len
        self.base = base
        self.sin, self.cos = self._compute_pos_emb()

    def _compute_pos_emb(self):
        from math import log
        pos = torch.arange(0, self.head_dim)
        div_term = torch.exp(torch.arange(0, self.head_dim, 2) * (- log(self.base) / self.head_dim))
        theta = torch.outer(pos, div_term)
        sin = theta.sin()
        cos = theta.cos()
        return sin, cos
    
    def forward(self, q: Tensor):
        bsz, seq_len, hidden_dim = q.shape
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        sin, cos = self.sin[:seq_len].unsqueeze(0).unsqueeze(0), self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        q_ = q.reshape(bsz, self.num_heads, seq_len, self.head_dim // 2, 2)
        q1 = q_[..., 0]
        q2 = q_[..., 1]
        o1 = q1 * cos - q2 * sin
        o2 = q2 * cos + q1 * sin
        out = torch.stack([o1, o2], dim=-1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        return out
    
class MultiHeadLatentAttn(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, down_dim=64, up_dim=128, rope_dim=32, dropout_rate=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.rope_dim = rope_dim
        self.dropout_rate = dropout_rate
        self.v_head_dim = up_dim // num_heads
        self.down_proj_q = nn.Linear(hidden_dim, down_dim)
        self.down_proj_kv = nn.Linear(hidden_dim, down_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)
        self.up_proj_k = nn.Linear(down_dim, up_dim)
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.proj_qr = nn.Linear(down_dim, rope_dim * num_heads)
        self.proj_kr = nn.Linear(hidden_dim, rope_dim * 1)
        self.rope_q = RotaryEmbedding(rope_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.v_head_dim * num_heads, hidden_dim)
        self.res_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X: Tensor, mask=None):
        bsz, seq_len, hidden_dim = X.shape
        c_kv = self.down_proj_kv(X)
        c_q = self.down_proj_q(X)
        k_c = self.up_proj_k(c_kv)
        v_c = self.up_proj_v(c_kv)
        q_c = self.up_proj_q(c_q)
        q_r = self.rope_q(self.proj_qr(c_q))
        k_r = self.rope_k(self.proj_kr(X))
        q_c = q_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        q = torch.cat([q_c, q_r], dim=-1)
        k_c = k_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        k_r = k_r.repeat(1, self.num_heads, 1, 1)
        k = torch.cat([k_c, k_r], dim=-1)
        attn_scores: Tensor = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5 + self.rope_dim ** 0.5)
        if mask is not None:
            attn_scores.masked_fill_(mask, value=-1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        v_c = v_c.reshape(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        o1 = torch.matmul(attn_weights, v_c).transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.res_dropout(self.fc(o1))
        return out
        
        