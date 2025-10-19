import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F

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

    def forward(self, x: Tensor, encoder_output: Tensor = None, attn_mask=None):
        bsz, tgt_seq_len, _ = x.shape
        if encoder_output is None: encoder_output = x
        src_seq_len = encoder_output.shape[1]
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(encoder_output)
        v: Tensor = self.v_proj(encoder_output)
        q = q.reshape(bsz, tgt_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, src_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, src_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim ** 0.5
        if attn_mask is not None: attn_score += attn_mask * -1e9
        attn_probs = F.softmax(attn_score, dim=-1)
        output_mid = torch.matmul(attn_probs, v)
        output_mid = output_mid.transpose(1, 2).reshape(bsz, tgt_seq_len, self.hidden_dim)
        output = self.o_proj(output_mid)
        return output

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x: Tensor):
        output = self.embedding(x)
        return output
    
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (- math.log(10000) / hidden_dim))
        pe = torch.zeros(max_seq_len, hidden_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # pe [1, max_seq_len, hidden_dim]

    def forward(self, x: Tensor):
        seq_len = x.shape[1]
        x += self.pe[:, :seq_len, :]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, mha_dp, ffn_dp, ffn_sz, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mha = MultiHeadAttention(hidden_dim, num_heads)
        self.dp1 = nn.Dropout(mha_dp)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_sz),
            nn.ReLU(),
            nn.Linear(ffn_sz, hidden_dim)
        )
        self.dp2 = nn.Dropout(ffn_dp)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, attn_mask=None):
        mha_o = self.mha(x, attn_mask = attn_mask)
        mha_o = self.dp1(mha_o)
        out1 = self.ln1(mha_o + x)

        ffn_o = self.ffn(out1)
        ffn_o = self.dp2(ffn_o)
        out2 = self.ln2(out1 + ffn_o)

        return out2
    
class Encoder(nn.Module):
    def __init__(self, 
                hidden_dim, num_heads, 
                mha_dp, ffn_dp, 
                ffn_sz, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, mha_dp, ffn_dp, ffn_sz)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, attn_mask):
        for layer in self.layers: 
            x = layer(x, attn_mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, 
                hidden_dim, num_heads, 
                self_mha_dp, cross_mha_dp, ffn_dp, 
                ffn_sz, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.dp1 = nn.Dropout(self_mha_dp)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.cross_mha =MultiHeadAttention(hidden_dim, num_heads)
        self.dp2 = nn.Dropout(cross_mha_dp)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_sz),
            nn.ReLU(),
            nn.Linear(ffn_sz, hidden_dim)
        )
        self.dp3 = nn.Dropout(ffn_dp)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, encoder_output: Tensor, self_attn_mask = None, cross_attn_mask = None):
        self_mha_o = self.self_mha(x, attn_mask = self_attn_mask)
        self_mha_o = self.dp1(self_mha_o)
        out1 = self.ln1(x + self_mha_o)

        cross_mha_o = self.cross_mha(out1, encoder_output, attn_mask = cross_attn_mask)
        cross_mha_o = self.dp2(cross_mha_o)
        out2 = self.ln2(out1 + cross_mha_o)
        
        ffn_o = self.ffn(out2)
        ffn_o = self.dp3(ffn_o)
        out3 = self.ln3(out2 + ffn_o)

        return out3
    
class Decoder(nn.Module):
    def __init__(self, 
                hidden_dim, num_heads, 
                self_mha_dp, cross_mha_dp, ffn_dp, 
                ffn_sz, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, self_mha_dp, cross_mha_dp, ffn_dp, ffn_sz)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor, encoder_out: Tensor, self_attn_mask = None, cross_attn_mask = None):
        for layer in self.layers:
            x = layer(x, encoder_out, self_attn_mask, cross_attn_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, 
                vocab_size, max_seq_len,
                hidden_dim, num_heads, 
                encoder_mha_dp, encoder_ffn_dp, encoder_ffn_sz, 
                decoder_self_mha_dp, decoder_cross_mha_dp, decoder_ffn_dp, decoder_ffn_sz, 
                encoder_num_layers, decoder_num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_len, hidden_dim)
        self.encoder = Encoder(hidden_dim, num_heads, 
                            encoder_mha_dp, encoder_ffn_dp, encoder_ffn_sz, 
                            encoder_num_layers)
        self.decoder = Decoder(hidden_dim, num_heads, 
                            decoder_self_mha_dp, decoder_cross_mha_dp, decoder_ffn_dp, decoder_ffn_sz, 
                            decoder_num_layers)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask = None, tgt_mask = None, src_tgt_mask = None):
        # x [bsz, seq_len]
        src_emb = self.positional_embedding(self.token_embedding(src))
        tgt_emb = self.positional_embedding(self.token_embedding(tgt))

        encoder_output = self.encoder(src_emb, src_mask)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_tgt_mask)

        logits = self.output_linear(decoder_output)
        print(f"logits shape : {logits.shape}")
        
        probs = F.softmax(logits, dim=-1)
        print(f"probs shape : {probs.shape}")

        return probs



def transformer_test():
    
    bsz, vocab_size, max_seq_len, src_seq_len, tgt_seq_len, hidden_dim, num_heads = 6, 512, 64, 5, 7, 16, 8
    encoder_mha_dp, encoder_ffn_dp, encoder_ffn_sz, encoder_num_layers = 0.1, 0.1, 4 * hidden_dim, 10
    decoder_self_mha_dp, decoder_cross_mha_dp, decoder_ffn_dp, decoder_ffn_sz, decoder_num_layers = 0.1, 0.1, 0.1, 4 * hidden_dim, 10
    
    src = torch.randint(0, vocab_size, (bsz, src_seq_len))
    tgt = torch.randint(0, vocab_size, (bsz, tgt_seq_len))

    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()

    transformer = Transformer(vocab_size, max_seq_len, 
                            hidden_dim, num_heads,
                            encoder_mha_dp=encoder_mha_dp,
                            encoder_ffn_dp=encoder_ffn_dp,
                            encoder_ffn_sz=encoder_ffn_sz,
                            encoder_num_layers=encoder_num_layers,
                            decoder_self_mha_dp=decoder_self_mha_dp,
                            decoder_cross_mha_dp=decoder_cross_mha_dp,
                            decoder_ffn_dp=decoder_ffn_dp,
                            decoder_ffn_sz=decoder_ffn_sz,
                            decoder_num_layers=decoder_num_layers)
    
    output = transformer(src, tgt, tgt_mask = tgt_mask)

    return output

if __name__ == "__main__":
    transformer_test()