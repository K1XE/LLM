import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hidden_state: Tensor, past_key_value = None, casual_mask: Tensor = None, pad_mask: Tensor = None, use_cache: bool = False):
        bsz, seq_len, _ = hidden_state.shape
        q: Tensor = self.q_proj(hidden_state) # [bsz, 1, hidden_dim]
        k: Tensor = self.k_proj(hidden_state)
        v: Tensor = self.v_proj(hidden_state)
        q = q.reshape(bsz, seq_len, self.num_heads, -1).permute(0, 2, 1, 3) # [bsz, num_heads, 1, head_dim]
        k = k.reshape(bsz, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        if past_key_value is not None:
            past_key, past_val = past_key_value
            k = torch.cat([past_key, k], dim=2) # [bsz, num_heads, seq_len, head_dim]
            v = torch.cat([past_val, v], dim=2)
        new_past_key_value = (k, v) if use_cache else None
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if casual_mask is not None:
            attn_score += casual_mask * -1e9
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            attn_score += pad_mask * -1e9
        attn_probs = F.softmax(attn_score, dim=-1) # [bsz, num_heads, 1, seq_len]
        output_mid = torch.matmul(attn_probs, v) # [bsz, num_heads, 1, hidden_dim]
        output_mid = output_mid.permute(0, 2, 1, 3).reshape(bsz, seq_len, -1) # [bsz, seq_len, hidden_dim]
        output = self.o_proj(output_mid)
        return (output, new_past_key_value) if use_cache else output

def MHA_with_kv_cache_test():
    bsz, seq_len, num_heads, hidden_dim = 8, 4, 2, 32
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mha_with_kv_cache = MultiHeadAttentionWithKVCache(hidden_dim, num_heads)
    past_key_value = None
    output = []
    for i in range(seq_len):
        cur_input = hidden_state[:, i:i + 1, :]
        cur_casual_mask = casual_mask[i:i + 1, :i + 1]
        output_step, past_key_value = mha_with_kv_cache(cur_input, past_key_value, cur_casual_mask, use_cache=True)
        output.append(output_step)
    output = torch.cat(output, dim=1)
    return output

if __name__ == "__main__":
    MHA_with_kv_cache_test()