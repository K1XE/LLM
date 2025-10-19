import torch
import torch.nn.functional as F
from torch import nn, Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, context: Tensor = None, self_attention_mask = None):
        if context is None: context = x
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q = q.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if self_attention_mask is not None: attn_score += self_attention_mask * -1e9
        attn_weights = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_weights, v)
        output_mid = output_mid.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)
        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, self_attn_dp = 0.1, encoder_decoder_dp = 0.1, ffn_dp = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.dp1 = nn.Dropout(self_attn_dp)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.encoder_decoder_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.dp2 = nn.Dropout(encoder_decoder_dp)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.dp3 = nn.Dropout(ffn_dp)
        self.ln3 = nn.LayerNorm(hidden_dim)
    def forward(self, x: Tensor, encoder_out: Tensor, self_attn_mask = None, encoder_decoder_mask = None):
        self_attn_o = self.self_attn(x, x, self_attn_mask)
        self_attn_o = self.dp1(self_attn_o)
        out1 = self.ln1(x + self_attn_o)

        encoder_decoder_attn_o = self.encoder_decoder_attn(out1, encoder_out, encoder_decoder_mask)
        encoder_decoder_attn_o = self.dp2(encoder_decoder_attn_o)
        out2 = self.ln2(out1 + encoder_decoder_attn_o)

        ffn_o = self.ffn(out2)
        ffn_o = self.dp3(ffn_o)
        out3 = self.ln3(out2 + ffn_o)

        return out3

def decoder_layer_test():
    bsz, seq_len, hidden_dim, num_heads = 6, 5, 16, 4
    x = torch.randn(bsz, seq_len, hidden_dim)
    encoder_out = torch.randn(bsz, seq_len, hidden_dim)
    self_attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    decoder = DecoderLayer(hidden_dim, num_heads, self_attn_dp=0.05, encoder_decoder_dp=0.05, ffn_dp=0.05)
    output = decoder(x, encoder_out, self_attention_mask)
    print(output.shape)
    return output

if __name__ == "__main__":
    decoder_layer_test()