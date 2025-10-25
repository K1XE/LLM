import torch
import math
from torch import nn, Tensor


def get_rope_sincos(head_dim, max_len, device=None, dtype=torch.float32):
    assert head_dim % 2 == 0
    pos = torch.arange(max_len, device=device, dtype=dtype)
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=device) * (-math.log(10_000) / head_dim)
    ).to(dtype)
    freqs = torch.outer(pos, div_term)
    sin = torch.sin(freqs).to(device)
    cos = torch.cos(freqs).to(device)
    return sin, cos


def apply_rope(X: Tensor, sin: Tensor, cos: Tensor):
    bsz, num_heads, seq_len, head_dim = X.shape
    assert head_dim % 2 == 0
    dtype = X.dtype
    device = X.device
    sin = sin.to(dtype).to(device)
    cos = cos.to(dtype).to(device)
    X_ = X.reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
    X_even = X_[..., 0]
    X_odd = X_[..., 1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    out_even = X_even * cos - X_odd * sin
    out_odd = X_even * sin + X_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1).reshape(
        bsz, num_heads, seq_len, head_dim
    )
    return out


def apply_rope_complex(X: Tensor, sin: Tensor, cos: Tensor):
    bsz, num_heads, seq_len, head_dim = X.shape
    assert head_dim % 2 == 0
    dtype = X.dtype
    device = X.device
    sin = sin[:seq_len].to(dtype).to(device)
    cos = cos[:seq_len].to(dtype).to(device)
    X_ = X.reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
    X_c = torch.view_as_complex(X_.to(torch.float32))
    mult = torch.complex(cos, sin)
    mult = mult.unsqueeze(0).unsqueeze(0)
    out_c = X_c * mult
    out_real = torch.view_as_real(out_c).to(dtype)
    out = out_real.reshape(bsz, num_heads, seq_len, head_dim)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.half
    bsz, num_heads, seq_len, head_dim = 4, 2, 7, 6
    max_len = 10
    sin, cos = get_rope_sincos(head_dim=head_dim, max_len=max_len, device=device)
    X = torch.randn(bsz, num_heads, seq_len, head_dim).to(device=device, dtype=dtype)
    out1 = apply_rope(X, sin, cos)
    out2 = apply_rope_complex(X, sin, cos)
    if not torch.allclose(out1.to(torch.float32), out2.to(torch.float32), atol=1e-2, rtol=1e-2):
        maxdiff = (out1.to(torch.float32) - out2.to(torch.float32)).abs().max()
        print(f"两种实现最大误差: {maxdiff}")
    else: print("all right")

if __name__ == "__main__":
    o = main()
