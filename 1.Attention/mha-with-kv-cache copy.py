from ast import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules import padding

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                hidden_state: Tensor, 
                past_key_value: Tensor = None, 
                casual_mask: Tensor = None, 
                padding_mask: Tensor = None, 
                use_cache: bool = False):
        bsz = hidden_state.shape[0]
        q: Tensor = self.q_proj(hidden_state) # [bsz, 1, hidden_dim]
        k: Tensor = self.k_proj(hidden_state)
        v: Tensor = self.v_proj(hidden_state)
        
        q = q.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2) # [bsz, num_heads, 1, head_dim]
        k = k.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_val = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_val, v], dim=2)
        nxt_key_value = k, v
        attn_score = torch.matmul(q, k.transpose(-1, -2)) \
                    / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) # [bsz, num_heads, 1, seq_len]
        if casual_mask is not None:
            attn_score += casual_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_score += padding_mask * -1e9
        attn_probs = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_probs, v) # [bsz, num_heads, 1, head_dim]
        output_mid = output_mid.transpose(1, -1).reshape(bsz, -1, self.hidden_dim)
        output = self.o_proj(output_mid)
        return (output, nxt_key_value) if use_cache else output
    
def MHA_with_kv_cache_test():
    bsz, seq_len, hidden_dim, num_heads = 8, 4, 64, 4
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mha_with_kv_cache = MultiHeadAttentionWithKVCache(hidden_dim, num_heads)
    past_key_value_cache = None
    use_cache = True
    output = []
    for i in range(seq_len):
        cur_input = hidden_state[:, i:i + 1, :]
        cur_casual_mask = casual_mask[i:i + 1, :i + 1]
        output_i, past_key_value_cache = mha_with_kv_cache(cur_input, past_key_value_cache, cur_casual_mask, use_cache=use_cache)
        output.append(output_i)
    output = torch.cat(output, dim=1)
    print("Input shape", hidden_state.shape)
    print("Output shape", output.shape)
    return output


if __name__ == "__main__":
    o = MHA_with_kv_cache_test()
    