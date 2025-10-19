import torch
import torch.nn.functional as F
from torch import nn, Tensor, unsqueeze

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_group, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_group = num_group
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_group * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_group * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, hidden_state: Tensor, casual_mask: Tensor = None, padding_mask: Tensor = None):
        bsz, seq_len, _ = hidden_state.shape
        q: Tensor = self.q_proj(hidden_state) # [bsz, seq_len, hidden_dim]
        k: Tensor = self.k_proj(hidden_state) # [bsz, seq_len, num_group * head_dim]
        v: Tensor = self.v_proj(hidden_state)
        def split_head(x: Tensor, num_group = None):
            if num_group is None:
                x = x.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2) # [bsz, num_heads, seq_len, head_dim]
            else:
                x = x.reshape(bsz, seq_len, num_group, -1).transpose(1, 2) # [bsz, num_group, seq_len, head_dim]
                # [bsz, num_heads. seq_len, head_dim]
                x = x.unsqueeze(2).expand(bsz, num_group, self.num_heads // num_group, seq_len, -1).reshape(bsz, -1, seq_len, self.head_dim)
            return x
        q = split_head(q)
        k = split_head(k, self.num_group)
        v = split_head(v, self.num_group)
        # [bsz, num_heads, seq, seq]
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if casual_mask is not None: attn_score += casual_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1),unsqueeze(1)
            attn_score += padding_mask * -1e9
        
        attn_probs = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_probs, v) # [bsz, num_heads, seq_len, head_dim]
        output = output_mid.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output

def GQA_test():
    bsz, seq_len, hidden_dim, num_heads, num_group = 8, 5, 64, 16, 2
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    GQA = GroupQueryAttention(hidden_dim, num_heads, num_group)
    output = GQA(hidden_state, casual_mask)
    return output
    
if __name__ == "__main__":
    GQA_test()