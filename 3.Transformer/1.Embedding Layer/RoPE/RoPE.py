# https://www.jackzhu.top/2024/04/29/%E6%B7%98%E5%A4%A9%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A0%E9%9D%A2%E8%AF%95%E5%88%86%E4%BA%AB/index.html#:~:text=linears%5B%2D1%5D(x)-,%E6%89%8B%E6%92%95%20rope%20%E7%AE%97%E6%B3%95,-PLAINTEXT

import torch
import math
from torch import Tensor, nn


def SinusoidalPositionalEmbedding(bsz, num_heads, max_len, output_dim, device):
    # (max_len, 1)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim // 2)
    theta = torch.exp(
        torch.arange(0, output_dim, 2, dtype=torch.float)
        * (-math.log(10_000) / output_dim)
    )
    # (max_len, output_dim // 2)
    embeddings = pos * theta
    # (max_len, output_dim // 2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    # (bsz, head, max_len, output_dim//2, 2)
    embeddings = embeddings.expand(bsz, num_heads, -1, -1, -1)
    # (bs, head, max_len, output_dim)
    embeddings = embeddings.reshape(bsz, num_heads, max_len, output_dim)
    embeddings = embeddings.to(device)
    return embeddings


def apply_RoPE(q: Tensor, k: Tensor):
    bsz, num_heads, seq_len, head_dim = q.shape
    pos_emb: Tensor = SinusoidalPositionalEmbedding(
        bsz=bsz, num_heads=num_heads, max_len=seq_len, output_dim=head_dim
    )
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape) # reshape后就是正负交替了
    q = q * cos_pos + q2 * sin_pos
    
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos
    return q, k
    
