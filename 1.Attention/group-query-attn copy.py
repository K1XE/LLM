import torch
import torch.nn.functional as F
from torch import nn, Tensor

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, group_num, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.group_num = group_num
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.group_num * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.group_num * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, hidden_state: Tensor, casual_mask: Tensor = None, padding_mask: Tensor = None):
        bsz, seq_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)
        def split_head(X: Tensor, group_num = None):
            if group_num is None:
                X = X.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
            else:
                X = X.reshape(bsz, seq_len, group_num, -1).transpose(1, 2)
                X = X.unsqueeze(2).expand(bsz, group_num, self.num_heads // group_num, seq_len, self.head_dim)
                X = X.reshape(bsz, -1, seq_len, self.head_dim)
            return X
        q = split_head(q)
        k = split_head(k, self.group_num)
        v = split_head(v, self.group_num)

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if casual_mask is not None: attn_score += casual_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_score += padding_mask * -1e9
        attn_weight = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_weight, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)
        return output

def GQA_test():
    bsz, seq_len, hidden_dim, group_num, num_heads = 8, 5, 64, 2, 16
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    gqa = GroupQueryAttention(hidden_dim, group_num, num_heads)
    output = gqa(hidden_state, casual_mask)
    return output

if __name__ == "__main__":
    GQA_test()