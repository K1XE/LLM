import torch
from torch import nn, Tensor
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x: Tensor, encoder_output: Tensor, self_attn_mask = None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)
        def split_head(x: Tensor, num_groups=None):
            if num_groups is None:
                x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                x = x.reshape(bsz, seq_len, num_groups, self.head_dim).transpose(1, 2).unsqueeze(2).expand(-1, -1, self.num_heads // num_groups, -1, -1).reshape(bsz, -1, seq_len, self.head_dim)
            return x
        q = split_head(q)
        k = split_head(k, self.num_groups)
        v = split_head(v, self.num_groups)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if self_attn_mask is not None: attn_scores += self_attn_mask * -1e9
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_mid = torch.matmul(attn_weights, v)
        output_mid = output_mid.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)
        return output
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, self_gqa_dp, cross_gqa_dp, ffn_dp, ffn_sz, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_gqa = GroupQueryAttention(hidden_dim, num_groups, num_heads)
        self.dp1 = nn.Dropout(self_gqa_dp)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.cross_gqa = GroupQueryAttention(hidden_dim, num_groups, num_heads)
        self.dp2 = nn.Dropout(cross_gqa_dp)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_sz),
            nn.ReLU(),
            nn.Linear(ffn_sz, hidden_dim)
        )
        self.dp3 = nn.Dropout(ffn_dp)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, encoder_output: Tensor, self_attn_mask=None):
        self_gqa_o = self.self_gqa(x, x, self_attn_mask)
        self_gqa_o = self.dp1(self_gqa_o)
        out1 = self.ln1(x + self_gqa_o)

        cross_gqa_o = self.cross_gqa(out1, encoder_output)
        cross_gqa_o = self.dp2(cross_gqa_o)
        out2 = self.ln2(out1 + cross_gqa_o)

        ffn_o = self.ffn(out2)
        ffn_o = self.dp3(ffn_o)
        out3 = self.ln3(out2 + ffn_o)
        
        return out3

class Deocder(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, self_gqa_dp, cross_gqa_dp, ffn_dp, ffn_sz, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_groups, num_heads, self_gqa_dp, cross_gqa_dp, ffn_dp, ffn_sz)
            for _ in range(num_layers)
        ])
    def forward(self, x: Tensor, encoder_output: Tensor, self_attn_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask)
        return x

def decoder_test():
    bsz, seq_len, hidden_dim, num_groups, num_heads = 6, 5, 16, 2, 4
    self_gqa_dp, cross_gqa_dp, ffn_dp = 0.05, 0.05, 0.05
    ffn_sz = 4 * hidden_dim
    num_layers = 10
    x = torch.randn(bsz, seq_len, hidden_dim)
    encoder_output = torch.randn(bsz, seq_len, hidden_dim)
    self_attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    decoder = Deocder(
        hidden_dim,
        num_groups,
        num_heads,
        self_gqa_dp,
        cross_gqa_dp,
        ffn_dp,
        ffn_sz,
        num_layers
    )
    output = decoder(x, encoder_output, self_attn_mask)
    print(output.shape)
    return output

if __name__ == "__main__":
    decoder_test()