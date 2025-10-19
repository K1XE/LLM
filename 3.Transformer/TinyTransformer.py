import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, X):
        emb = self.embedding(X)
        return emb  # [bsz, seq_len, hidden_dim]


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        # import pdb; pdb.set_trace()
        pos = torch.arange(0, max_len).unsqueeze(-1).to(torch.float32)
        div_term = torch.exp(
            -torch.arange(0, hidden_dim, 2) / hidden_dim * math.log(10000)
        )
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe: Tensor
        self.register_buffer("pe", pe)

    def forward(self, X: Tensor):
        # import pdb; pdb.set_trace()
        _, seq_len, _ = X.shape
        o = X + self.pe[:seq_len, :].unsqueeze(0).to(X.device)
        return o


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_groups, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_groups * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, encoder_output, causal_mask=None, padding_mask=None):
        bsz, tgt_seq_len, _ = X.shape
        q = self.q_proj(X)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)

        def split_heads(X: Tensor, num_groups=None):
            # X: [bsz, seq_len, hidden_dim] → [bsz, num_heads, seq_len, head_dim]
            L = X.shape[1]
            if num_groups is None:
                X = X.reshape(bsz, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            else:
                assert self.num_heads % num_groups == 0
                X = (
                    X.reshape(bsz, L, num_groups, self.head_dim)
                    .transpose(1, 2)
                    .unsqueeze(2)
                    .expand(-1, -1, self.num_heads // num_groups, -1, -1)
                    .reshape(bsz, -1, L, self.head_dim)
                )
            return X

        # import pdb; pdb.set_trace()

        # [bsz, num_heads, seq_len, head_dim]
        q = split_heads(q)
        k = split_heads(k, num_groups=self.num_groups)
        v = split_heads(v, num_groups=self.num_groups)
        attn_scores: Tensor = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        if causal_mask is not None:
            attn_scores.masked_fill_(
                causal_mask.to(attn_scores.device).unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
        if padding_mask is not None:
            attn_scores.masked_fill_(
                padding_mask[:, None, None, :].to(attn_scores.device), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_mid = (
            torch.matmul(attn_weights, v)
            .permute(0, 2, 1, 3)
            .reshape(
                bsz,
                tgt_seq_len,
                self.hidden_dim,
            )
        )
        o = self.o_proj(output_mid)
        return o


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, X):
        o = self.ffn(X)
        return o


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_groups,
        num_heads,
        dp_rate_gqa: float = 0.1,
        dp_rate_ffn: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gqa = GroupQueryAttention(
            hidden_dim=hidden_dim, num_groups=num_groups, num_heads=num_heads
        )
        self.pre_rmsnorm1 = nn.RMSNorm(hidden_dim)
        self.dp1 = nn.Dropout(dp_rate_gqa)
        self.ffn = FeedForwardNetwork(hidden_dim=hidden_dim)
        self.pre_rmsnorm2 = nn.RMSNorm(hidden_dim)
        self.dp2 = nn.Dropout(dp_rate_ffn)

    def forward(self, X):
        x_norm = self.pre_rmsnorm1(X)
        gqa_out = self.gqa(x_norm, x_norm)
        gqa_out = self.dp1(gqa_out) + X

        o_norm = self.pre_rmsnorm2(gqa_out)
        ffn_out = self.ffn(o_norm)
        out = self.dp2(ffn_out) + gqa_out
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_groups,
        num_heads,
        dp_rate_gqa: float = 0.1,
        dp_rate_cross: float = 0.1,
        dp_rate_ffn: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gqa = GroupQueryAttention(
            hidden_dim=hidden_dim, num_groups=num_groups, num_heads=num_heads
        )
        self.dp1 = nn.Dropout(dp_rate_gqa)
        self.pre_rmsnorm1 = nn.RMSNorm(hidden_dim)
        self.gqa_cross = GroupQueryAttention(
            hidden_dim=hidden_dim, num_groups=num_groups, num_heads=num_heads
        )
        self.dp2 = nn.Dropout(dp_rate_cross)
        self.pre_rmsnorm2 = nn.RMSNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim=hidden_dim)
        self.dp3 = nn.Dropout(dp_rate_ffn)
        self.pre_rmsnorm3 = nn.RMSNorm(hidden_dim)

    def forward(self, X, encoder_output, causal_mask=None, padding_mask=None):
        x_norm = self.pre_rmsnorm1(X)
        o1 = self.gqa(x_norm, x_norm, causal_mask, padding_mask)
        o1 = self.dp1(o1) + X
        o1_norm = self.pre_rmsnorm2(o1)
        o2 = self.gqa_cross(o1_norm, encoder_output)
        o2 = self.dp2(o2) + o1
        o2_norm = self.pre_rmsnorm3(o2)
        o3 = self.ffn(o2_norm)
        o3 = self.dp3(o3) + o2
        return o3


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_groups,
        num_heads,
        num_layers,
        dp_rate_gqa: float = 0.1,
        dp_rate_ffn: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoderlayers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim, num_groups, num_heads, dp_rate_gqa, dp_rate_ffn
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, X):
        for layer in self.encoderlayers:
            X = layer(X)
        return X


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_groups,
        num_heads,
        num_layers,
        dp_rate_gqa: float = 0.1,
        dp_rate_cross: float = 0.1,
        dp_rate_ffn: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decoderlayers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim,
                    num_groups,
                    num_heads,
                    dp_rate_gqa,
                    dp_rate_cross,
                    dp_rate_ffn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, X, encoder_output, causal_mask=None, padding_mask=None):
        for layer in self.decoderlayers:
            X = layer(X, encoder_output, causal_mask, padding_mask)
        return X


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_dim,
        num_groups,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        encoder_dp_rate_gqa=0.1,
        encoder_dp_rate_ffn=0.1,
        decoder_dp_rate_gqa=0.1,
        decoder_dp_rate_cross=0.1,
        decoder_dp_rate_ffn=0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.TokenEmbedding = TokenEmbedding(
            vocab_size=vocab_size, hidden_dim=hidden_dim
        )
        self.PositionalEncoding = PositionalEncoding(
            max_len=max_len, hidden_dim=hidden_dim
        )
        self.Encoder = Encoder(
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dp_rate_gqa=encoder_dp_rate_gqa,
            dp_rate_ffn=encoder_dp_rate_ffn,
        )
        self.Decoder = Decoder(
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dp_rate_gqa=decoder_dp_rate_gqa,
            dp_rate_cross=decoder_dp_rate_cross,
            dp_rate_ffn=decoder_dp_rate_ffn,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, src: Tensor, tgt: Tensor, causal_mask=None, padding_mask=None):
        # X -> [bsz, seq_len]
        encoder_pos_out = self.PositionalEncoding(self.TokenEmbedding(src))
        encoder_out = self.Encoder(encoder_pos_out)
        decoder_pos_out = self.PositionalEncoding(self.TokenEmbedding(tgt))
        hidden = self.Decoder(decoder_pos_out, encoder_out, causal_mask, padding_mask)
        logits = self.lm_head(hidden)
        return logits


def main():
    bsz, max_len, vocab_size = 5, 8, 15
    src_seq_len, tgt_seq_len = 5, 7
    hidden_dim, num_groups, num_heads = 16, 2, 8
    src = torch.randint(0, vocab_size, (bsz, src_seq_len))
    tgt = torch.randint(0, vocab_size, (bsz, tgt_seq_len))
    tgt_causal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).to(
        torch.bool
    )
    transformer = Transformer(
        vocab_size=vocab_size,
        max_len=max_len,
        hidden_dim=hidden_dim,
        num_groups=num_groups,
        num_heads=num_heads,
        num_encoder_layers=10,
        num_decoder_layers=10,
        encoder_dp_rate_gqa=0.15,
        encoder_dp_rate_ffn=0.15,
        decoder_dp_rate_gqa=0.15,
        decoder_dp_rate_cross=0.15,
        decoder_dp_rate_ffn=0.15,
    )
    output = transformer(src, tgt, tgt_causal_mask)
    print(output.shape)
    return output


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    o = main()
    print(o[0, :, :])
