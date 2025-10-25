import torch
from math import log
from torch import Tensor

def get_sincos(max_len, hidden_dim):
    pos = torch.arange(0, max_len).unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (- log(10000) / hidden_dim))
    theta = torch.outer(pos, div_term)
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    
    return sin, cos

def apply_rope(X: Tensor, max_len):
    bsz, num_heads, seq_len, head_dim = X.shape
    X = X.reshape(..., head_dim // 2, 2)
    X1 = X[..., 0]
    X2 = X[..., 1]
    sin, cos = get_sincos(max_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    out1 = X1 * cos - X2 * sin
    out2 = X2 * cos + X1 * sin
    out = torch.stack([out1, out2], dim=-1).reshape(bsz, num_heads, seq_len, head_dim)
    
    return out