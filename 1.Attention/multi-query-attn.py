import torch
from torch import nn, Tensor
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim: Tensor, num_heads: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, hidden_state: Tensor, casual_mask: Tensor = None, pad_mask: Tensor = None):
        bsz, seq_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)
        def split_head(x: Tensor, num_heads=None):
            # [bsz, seq, hidden_dim]
            bsz = x.shape[0]
            if num_heads is None: num_heads = self.num_heads
            return x.reshape(bsz, -1, num_heads, self.head_dim).transpose(1, 2)
        q = split_head(q) # [bsz, num_heads, seq_len, head_dim]
        k = split_head(k, 1) # [bsz, 1, seq_len, head_dim]
        v = split_head(v, 1) # [bsz, 1, seq_len, head_dim]
        attn_score = q @ k.transpose(-1, -2) / self.head_dim ** 0.5 # broadcase 
        if casual_mask is not None: attn_score += casual_mask * -1e9
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            attn_score += pad_mask * -1e9
        attn_probs = F.softmax(attn_score, dim=-1) # [bsz, num_heads, seq_len, seq_len]
        output_mid = attn_probs @ v # [bsz, num_heads, seq_len, head_dim]
        output_mid = output_mid.permute(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)
        return output

def mqa_test():
    bsz, seq_len, hidden_dim, num_heads = 64, 128, 1024, 4
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mqa = MultiQueryAttention(hidden_dim, num_heads)
    output = mqa(hidden_state, casual_mask)

if __name__ == "__main__":
    mqa_test()