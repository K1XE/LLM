import torch
from torch import nn, Tensor
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, X: Tensor, casual_mask = None):
        bsz, seq_len, hidden_dim = X.shape
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        def split_head(X: Tensor, num_groups = None):
            if num_groups is None:
                X = X.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                assert self.num_heads % num_groups == 0
                X = X.reshape(bsz, seq_len, num_groups, self.head_dim).transpose(1, 2).unsqueeze(2)
                X = X.expand(-1, -1, self.num_heads // num_groups, -1, -1).reshape(bsz, -1, seq_len, self.head_dim)
            return X
        q = split_head(q) # [bsz, num_head, seq, -1]
        k = split_head(k, self.num_groups) # [bsz, num_group * num_head // num_group, seq, -1]
        v = split_head(v, self.num_groups)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim ** 0.5
        if casual_mask is not None: attn_scores += casual_mask * -1e9
        attn_probs = F.softmax(attn_scores, dim=-1)
        output_mid = torch.matmul(attn_probs, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)
        return output

def main():
    bsz, seq_len, hidden_dim, num_groups, num_heads = 5, 4, 12, 2, 6
    X = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    gqa = GroupQueryAttention(hidden_dim, num_groups, num_heads)
    o = gqa(X, casual_mask)
    print(o.shape)

if __name__ == "__main__":
    main()