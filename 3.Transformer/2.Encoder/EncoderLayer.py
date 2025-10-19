import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, ffn_dim, dropout_prob=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gqa = GroupQueryAttention(hidden_dim, num_groups, num_heads)
        self.dropout_gqa = nn.Dropout(dropout_prob)
        self.layer_norm_gqa = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout_prob)
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
    def forward(self, x: Tensor, casual_mask = None, padding_mask = None):
        attn_output = self.gqa(x, casual_mask)
        attn_output = self.dropout_gqa(attn_output)
        out1 = self.layer_norm_gqa(attn_output + x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_ffn(ffn_output)
        out2 = self.layer_norm_ffn(ffn_output + out1)

        return out2

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_groups
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x: Tensor, casual_mask = None, padding_mask = None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        def split_heads(x: Tensor, num_groups=None):
            if num_groups is None:
                x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                x = x.reshape(bsz, seq_len, num_groups, -1).transpose(1, 2).unsqueeze(2) # [bsz, num_groups, 1, seq_len, -1]
                x = x.expand(-1, -1, self.num_heads // num_groups, -1, -1).reshape(bsz, -1, seq_len, self.head_dim)
            return x
        q = split_heads(q)
        k = split_heads(k, self.num_groups)
        v = split_heads(v, self.num_groups)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if casual_mask is not None:
            attn_score += casual_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_score += padding_mask * -1e9
        attn_weights = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_weights, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)
        return output
    

def encoder_test():
    bsz, seq_len, hidden_dim, num_heads, num_groups, ffn_dim = 6, 5, 16, 8, 4, 256
    x = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    encoder_layer = Encoder(hidden_dim, num_groups, num_heads, ffn_dim, dropout_prob=0.15)
    output = encoder_layer(x, casual_mask)
    print(output.shape)
    return output

if __name__ == "__main__":
    encoder_test()
