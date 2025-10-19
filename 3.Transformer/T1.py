import torch
from torch import nn, Tensor
import torch.nn.functional as F
from math import log


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, X: Tensor):
        X = self.emb(X)
        return X


class RoPE:
    @staticmethod
    def get_sin_cos(max_len, hidden_dim):
        pos = torch.arange(0, max_len).unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-log(10_000) / hidden_dim)
        )
        sin = torch.sin(pos * div_term)
        cos = torch.cos(pos * div_term)
        return sin, cos

    @staticmethod
    def apply_rope(X: Tensor, max_len):
        bsz, num_heads, seq_len, head_dim = X.shape
        assert head_dim % 2 == 0
        # X_ = X.reshape(bsz, num_heads, seq_len, head_dim // 2, 2)
        X_even = X[..., 0::2]
        X_odd = X[..., 1::2]
        device = X.device
        
        sin, cos = RoPE.get_sin_cos(max_len, head_dim)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(device)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(device)
        out_even = X_even * cos - X_odd * sin
        out_odd = X_even * sin + X_odd * cos
        # out = torch.stack([out_even, out_odd], dim=-1).reshape(
        #     bsz, num_heads, seq_len, head_dim
        # )
        out = torch.empty_like(X)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        
        return out


class GroupQueryAttention(nn.Module):
    def __init__(self, num_groups, num_heads, hidden_dim):
        super().__init__()
        self.num_groups = num_groups
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        X: Tensor,
        encoder_output: Tensor,
        max_len,
        causal_mask=None,
        padding_mask=None,
    ):
        bsz, tgt_seq_len, hidden_dim = X.shape
        q = self.q_proj(X)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)

        def split_heads(X: Tensor, num_groups=None):
            L = X.shape[1]
            if num_groups is None:
                X = X.reshape(bsz, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            else:
                X = X.reshape(bsz, L, num_groups, self.head_dim).transpose(1, 2)
                X = (
                    X.unsqueeze(2)
                    .expand(-1, -1, self.num_heads // num_groups, -1, -1)
                    .reshape(bsz, self.num_heads, L, self.head_dim)
                )
            return X

        q = split_heads(q)
        k = split_heads(k, self.num_groups)
        v = split_heads(v, self.num_groups)

        q = RoPE.apply_rope(q, max_len)
        k = RoPE.apply_rope(k, max_len)
        attn_scores: Tensor = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        if causal_mask is not None:
            attn_scores.masked_fill_(causal_mask, float("-inf"))
        if padding_mask is not None:
            attn_scores.masked_fill_(padding_mask[:, None, None, :], float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_mid = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .reshape(bsz, tgt_seq_len, hidden_dim)
        )
        out = self.o_proj(output_mid)
        return out


class EncoderLayer(nn.Module):
    def __init__(
        self, num_groups, num_heads, hidden_dim, dp1_rate, dp2_rate, ffn_dp_rate
    ):
        super().__init__()
        self.pre_rmsnorm1 = nn.RMSNorm(hidden_dim)
        self.gqa = GroupQueryAttention(num_groups, num_heads, hidden_dim)
        self.dp1 = nn.Dropout(dp1_rate)
        self.pre_rmsnorm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dp_rate),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dp2 = nn.Dropout(dp2_rate)

    def forward(self, X: Tensor, max_len):
        mid = self.pre_rmsnorm1(X)
        mid = self.gqa(mid, mid, max_len)
        o1 = X + self.dp1(mid)

        mid = self.pre_rmsnorm2(o1)
        mid = self.ffn(mid)
        o2 = o1 + self.dp2(mid)

        return o2


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_groups,
        num_heads,
        hidden_dim,
        dp1_rate,
        dp2_rate,
        ffn_dp_rate,
    ):
        super().__init__()
        self.encoderLayers = nn.ModuleList(
            [
                EncoderLayer(
                    num_groups=num_groups,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dp1_rate=dp1_rate,
                    dp2_rate=dp2_rate,
                    ffn_dp_rate=ffn_dp_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, X: Tensor, max_len):
        for layer in self.encoderLayers:
            X = layer(X, max_len)
        return X


class DecoderLayer(nn.Module):
    def __init__(
        self, num_groups, num_heads, hidden_dim, dp1_rate, dp2_rate, ffn_dp_rate
    ):
        super().__init__()
        self.pre_rmsnorm1 = nn.RMSNorm(hidden_dim)
        self.gqa = GroupQueryAttention(num_groups, num_heads, hidden_dim)
        self.dp1 = nn.Dropout(dp1_rate)

        self.pre_rmsnorm2 = nn.RMSNorm(hidden_dim)
        self.gqa_cross = GroupQueryAttention(num_groups, num_heads, hidden_dim)
        self.dp2 = nn.Dropout(dp2_rate)

        self.pre_rmsnorm3 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dp_rate),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dp3 = nn.Dropout(dp2_rate)

    def forward(
        self,
        X: Tensor,
        encoder_output: Tensor,
        max_len,
        causal_mask=None,
        padding_mask=None,
    ):
        mid = self.pre_rmsnorm1(X)
        mid = self.gqa(mid, mid, max_len, causal_mask=causal_mask)
        o1 = self.dp1(mid) + X

        mid = self.pre_rmsnorm2(o1)
        mid = self.gqa_cross(mid, encoder_output, max_len, padding_mask=padding_mask)
        o2 = self.dp2(mid) + o1

        mid = self.pre_rmsnorm3(o2)
        mid = self.ffn(mid)
        o3 = self.dp3(mid) + o2

        return o3


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_groups,
        num_heads,
        hidden_dim,
        dp1_rate,
        dp2_rate,
        ffn_dp_rate,
    ):
        super().__init__()
        self.decoderLayers = nn.ModuleList(
            [
                DecoderLayer(
                    num_groups, num_heads, hidden_dim, dp1_rate, dp2_rate, ffn_dp_rate
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        X: Tensor,
        encoder_output: Tensor,
        max_len,
        causal_mask=None,
        padding_mask=None,
    ):
        for layer in self.decoderLayers:
            X = layer(X, encoder_output, max_len, causal_mask, padding_mask)
        return X


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        num_encoder_layer,
        num_decoder_layer,
        num_groups,
        num_heads,
        hidden_dim,
        decoder_dp1_rate,
        decoder_dp2_rate,
        decoder_ffn_rate,
        encoder_dp1_rate,
        encoder_dp2_rate,
        encoder_ffn_rate,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_emb = TokenEmbedding(vocab_size, hidden_dim)

        self.encoder = Encoder(
            num_encoder_layer,
            num_groups,
            num_heads,
            hidden_dim,
            encoder_dp1_rate,
            encoder_dp2_rate,
            encoder_ffn_rate,
        )

        self.decoder = Decoder(
            num_decoder_layer,
            num_groups,
            num_heads,
            hidden_dim,
            decoder_dp1_rate,
            decoder_dp2_rate,
            decoder_ffn_rate,
        )

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_emb.emb.weight = self.lm_head.weight

    def forward(self, src: Tensor, tgt: Tensor, causal_mask=None, padding_mask=None):
        encoder_mid = self.token_emb(src)
        encoder_out = self.encoder(encoder_mid, self.max_len)

        decoder_mid = self.token_emb(tgt)
        decoder_out = self.decoder(
            decoder_mid, encoder_out, self.max_len, causal_mask, padding_mask
        )

        logits = self.lm_head(decoder_out)

        return logits


def main():
    bsz, max_len, vocab_size = 5, 16, 15
    src_seq_len, tgt_seq_len = 8, 10
    hidden_dim, num_groups, num_heads = 32, 2, 8
    num_encoder_layers, num_decoder_layers = 2, 2
    src = torch.randint(0, vocab_size, (bsz, src_seq_len))
    tgt = torch.randint(0, vocab_size, (bsz, tgt_seq_len))
    tgt_causal_mask = torch.triu(
        torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool), diagonal=1
    )
    src_padding_mask = torch.zeros(bsz, src_seq_len, dtype=torch.bool)
    src_padding_mask[:, -2:] = True

    model = Transformer(
        vocab_size=vocab_size,
        max_len=max_len,
        num_encoder_layer=num_encoder_layers,
        num_decoder_layer=num_decoder_layers,
        num_groups=num_groups,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        decoder_dp1_rate=0.1,
        decoder_dp2_rate=0.1,
        decoder_ffn_rate=0.1,
        encoder_dp1_rate=0.1,
        encoder_dp2_rate=0.1,
        encoder_ffn_rate=0.1,
    )
    
    logits = model(src, tgt, tgt_causal_mask, src_padding_mask)
    print(logits.shape)


if __name__ == "__main__":
    main()
