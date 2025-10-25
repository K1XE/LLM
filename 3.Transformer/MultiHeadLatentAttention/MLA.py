import torch
from torch import nn, Tensor


class RotaryEmbedding(nn.Module):
    def __init__(
        self, d_model, num_heads, base=10000, max_len=512, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.base = base
        self.max_len = max_len
        self.cos_pos_cache, self.sin_pos_cache = self._compute_pos_emb()

    def _compute_pos_emb(self):
        from math import log

        theta = torch.exp(
            torch.arange(0, self.head_dim, 2) * (-log(self.base) / self.head_dim)
        )
        pos = torch.arange(0, self.max_len)
        pos_emb = torch.outer(pos, theta)
        cos = pos_emb.sin().repeat_interleave(2, dim=-1)
        sin = pos_emb.cos().repeat_interleave(2, dim=-1)
        return cos, sin

    def forawrd(self, q: Tensor):
        bsz, q_len = q.shape[0], q.shape[1]
        self.cos_pos = self.cos_pos_cache[:q_len]
        self.sin_pos = self.sin_pos_cache[:q_len]
        q = q.reshape(bsz, q_len, self.num_heads, -1).transpose(1, 2)
        self.cos_pos = self.cos_pos.repeat(
            bsz, self.num_heads, *([1] * len(self.cos_pos.shape))
        )
        self.sin_pos = self.sin_pos.repeat(
            bsz, self.num_heads, *([1] * len(self.sin_pos.shape))
        )
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(bsz, self.num_heads, q_len, -1)

        r_q = q * self.cos_pos + q2 * self.sin_pos

        return r_q


class MLA(nn.Module):
    def __init__(
        self,
        d_model=512,
        down_dim=128,
        up_dim=256,
        num_heads=8,
        rope_head_dim=26,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = up_dim // num_heads

        self.down_proj_kv = nn.Linear(d_model, down_dim)
        self.up_proj_k = nn.Linear(down_dim, up_dim)
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.down_proj_q = nn.Linear(d_model, down_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)
        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        self.proj_kr = nn.Linear(d_model, rope_head_dim * 1)
        self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_heads * self.v_head_dim, d_model)
        self.res_dropout = nn.Dropout(dropout_prob)

    def forward(self, h: Tensor, mask=None):
        bsz, seq_len, _ = h.shape
        c_t_kv = self.down_proj_kv(h)
        k_t_c = self.up_proj_k(c_t_kv)
        v_t_c = self.up_proj_v(c_t_kv)
        c_t_q = self.down_proj_q(h)
        q_t_c = self.up_proj_q(c_t_q)

        q_t_r = self.rope_q(self.proj_qr(c_t_q))
        k_t_r = self.rope_k(self.proj_kr(h))

        q_t_c = q_t_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        q = torch.cat([q_t_c, q_t_r], dim=-1)
        k_t_c = k_t_c.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)

        k_t_r = k_t_r.repeat(1, self.num_heads, 1, 1)
        k = torch.cat([k_t_c, k_t_r], dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (
            self.num_heads**0.5 + self.rope_head_dim**0.5
        )
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        v_t_c = v_t_c.reshape(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(
            1, 2
        )
        output = torch.matmul(attn_weights, v_t_c)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.fc(output)
        output = self.res_dropout(output)
        return output
