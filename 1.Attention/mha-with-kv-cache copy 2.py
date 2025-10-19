import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MHAwithKVcache(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, X: Tensor, past_kv=None, causal_mask=None, pad_mask=None, use_cache=False
    ):
        bsz, seq_len, _ = X.shape
        q = (
            self.q_proj(X)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(X)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(X)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        if use_cache:
            new_kv = (k, v)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        if causal_mask is not None:
            attn_scores += -1e9 * causal_mask
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            attn_scores += -1e9 * pad_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        output_mid = (
            torch.matmul(attn_probs, v)
            .transpose(1, 2)
            .reshape(bsz, seq_len, self.hidden_dim)
        )
        output = self.o_proj(output_mid)
        return (output, new_kv) if use_cache else output


def main():
    bsz, seq_len, hidden_dim, num_heads = 5, 6, 16, 8
    X = torch.randn(bsz, seq_len, hidden_dim)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    past_kv = None
    use_cache = True
    mhakv = MHAwithKVcache(hidden_dim, num_heads)
    output = []
    for i in range(seq_len):
        cur_x = X[:, i : i + 1, :]
        cur_mask = causal_mask[i : i + 1, : i + 1]
        output_step, past_kv = mhakv(cur_x, past_kv, cur_mask, use_cache=use_cache)
        output.append(output_step)
    o = torch.cat(output, dim=1)
    return o


if __name__ == "__main__":
    o = main()
    print(o.shape)
