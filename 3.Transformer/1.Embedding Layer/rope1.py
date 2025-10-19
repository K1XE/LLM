import torch
from torch import nn, Tensor
from math import log

def get_sin_cos(max_len, head_dim):
    pos = torch.arange(max_len).unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, head_dim, 2) * (-log(10000) / head_dim))
    sin = torch.sin(pos * div_term)
    cos = torch.cos(pos * div_term)
    return sin, cos

def apply_rope(X: Tensor, max_len):
    bsz, num_head, seq_len, head_dim = X.shape
    sin, cos = get_sin_cos(max_len, head_dim)
    X_ = X.reshape(bsz, num_head, seq_len, head_dim // 2, 2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    X_even = X_[..., 0]
    X_odd = X_[..., 1]
    out_even = X_even * cos - X_odd * sin
    out_odd = X_even * sin + X_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1).reshape(bsz, num_head, seq_len, head_dim)
    return out