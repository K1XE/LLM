from torch import nn, Tensor
import torch
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, casual_mask = None, padding_mask = None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        def split_heads(x: Tensor, num_groups = None):
            if num_groups is None:
                x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                x = x.reshape(bsz, seq_len, self.num_groups, self.head_dim).transpose(1, 2).unsqueeze(2)
                x = x.expand(-1, -1, self.num_heads // num_groups, -1, -1).reshape(bsz, -1, seq_len, self.head_dim)
            return x
        q = split_heads(q)
        k = split_heads(k, self.num_groups)
        v = split_heads(v, self.num_groups)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if casual_mask is not None: attn_score += casual_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_score += padding_mask * -1e9
        attn_weights = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_weights, v)
        output = output_mid.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, gqa_dropout=0.1, ffn_dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gqa = GroupQueryAttention(hidden_dim, num_groups, num_heads)
        self.dp1 = nn.Dropout(gqa_dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.dp2 = nn.Dropout(ffn_dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
    def forward(self, x: Tensor, casual_mask = None, padding_mask = None):
        gqa_o = self.gqa(x, casual_mask, padding_mask)
        gqa_o = self.dp1(gqa_o)
        out1 = self.ln1(x + gqa_o)

        ffn_o = self.ffn(out1)
        ffn_o = self.dp2(ffn_o)
        out2 = self.ln2(out1 + ffn_o)
        return out2
    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, gqa_dropout, ffn_dropout, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_groups, num_heads, gqa_dropout, ffn_dropout) for _ in range(num_layers)
        ])
    def forward(self, x: Tensor, casual_mask = None, padding_mask = None):
        for layer in self.layers:
            x = layer(x, casual_mask)
        return x
    
def encoder_test():
    bsz, seq_len, hidden_dim, num_groups, num_heads, gqa_dropout, ffn_dropout = 6, 5, 16, 4, 8, 0.05, 0.05
    num_layers = 10
    x = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    encoder = Encoder(hidden_dim, num_groups, num_heads, gqa_dropout, ffn_dropout, num_layers)
    output = encoder(x, casual_mask)
    print(output.shape)
    return output

if __name__ == "__main__":
    encoder_test()

