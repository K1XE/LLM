import torch
from torch import nn, Tensor
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super.__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, X: Tensor):
        # [bsz, seq_len]
        X = self.emb(X)
        
        return X
    
class RoPE:
    @staticmethod
    def get_sin_cos(max_len, hidden_dim):
        from math import log
        pos = torch.arange(0, max_len)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (- log(10000) / hidden_dim))
        theta = torch.outer(pos.unsqueeze(-1), div_term)
        sin, cos = torch.sin(theta), torch.cos(theta)
        return sin, cos
    
    @staticmethod
    def apply_rope(X: Tensor, max_len):
        bsz, num_heads, seq_len, head_dim = X.shape
        X = X.reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
        X1 = X[..., 0]
        X2 = X[..., 1]
        sin, cos = RoPE.get_sin_cos(max_len, head_dim)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        out1 = X1 * cos - X2 * sin
        out2 = X2 * cos + X1 * sin
        out = torch.stack([out1, out2], dim=-1).reshape(bsz, num_heads, seq_len, head_dim)
        return out
    
class GroupQueryAttn(nn.Module):
    def __init__(self, num_heads, num_groups, hidden_dim):
        super.__init__()
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X: Tensor):
        bsz, seq_len, hidden_dim = X.shape
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        def split_heads(X: Tensor, num_groups=None):
            if num_groups is None:
                X = X.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                X = X.reshape(bsz, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
                X = X.unsqueeze(2).expand(-1, -1, self.num_heads // self.num_groups, -1, -1).reshape(bsz, -1, seq_len, self.head_dim)
                
            return X
        q = split_heads(q)
        k = split_heads(k, self.num_groups)
        v = split_heads(v, self.num_groups)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        mid = (attn_weights @ v).transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.o_proj(mid)
        
        return out
