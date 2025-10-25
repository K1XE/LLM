import torch

def get_sin_cos(max_len, hidden_dim):
    from math import log
    pos = torch.arange(0, max_len)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (- log(10_000) / hidden_dim))
    theta = torch.outer(pos, div_term)
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    return sin, cos

from torch import Tensor
def apply_rope(X: Tensor, max_len):
    bsz, num_heads, seq_len, head_dim = X.shape
    sin, cos = get_sin_cos(max_len, head_dim)
    X1 = X[..., :head_dim // 2]
    X2 = X[..., head_dim // 2:]
    sin = sin[None, None, :seq_len, ...]
    cos = cos[None, None, :seq_len, ...]
    o1 = X1 * cos - X2 * sin
    o2 = X2 * cos + X1 * sin
    o = torch.stack([o1, o2], dim=-1)
    o = o.reshape(bsz, num_heads, seq_len, head_dim)
    print(o.shape)
    return o

def main():
    X = torch.randn(5, 2, 3, 8)
    apply_rope(X, 10)
    
if __name__ == "__main__":
    main()