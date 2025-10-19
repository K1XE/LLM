import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    MLA（Multi-Head Latent Attention，多头潜变量注意力）的简化实现。

    设计动机（与传统 MHA/GQA/MQA 的区别）：
    - 传统做法缓存的是展开到每个 head 的 K/V（形如 [B, H, T, D_h]），显存占用与 heads 成正比。
    - MLA 将 K/V 先压缩到较小的潜变量维度 latent_dim（形如 [B, T, D_latent]）用于缓存，
      需要时再通过一个共享的线性变换“解码/展开”成每个 head 的 K/V（形如 [B, H, T, D_h]）。
    - 这样做可以显著降低 KV cache 的内存占用（与 D_latent 成正比，而不是 H×D_h）。

    本实现要点：
    - 投影：Q -> [B, T, H*D_h]；K_latent/V_latent -> [B, T, D_latent]
    - 展开：使用线性变换将 K_latent/V_latent 解码为各个 head 的 K/V。
    - KV Cache：仅缓存 K_latent/V_latent；按需解码为 per-head K/V。
    - 支持因果掩码与 padding 掩码。
    """

    def __init__(self, hidden_dim: int, num_heads: int, latent_dim: int) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.latent_dim = latent_dim

        # Q 为常规多头展开尺寸；K/V 先压缩到 latent 维度
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.v_latent_proj = nn.Linear(hidden_dim, latent_dim)

        # 将潜变量解码/展开为每个 head 的 K/V（共享展开矩阵）
        self.k_expand = nn.Linear(latent_dim, num_heads * self.head_dim)
        self.v_expand = nn.Linear(latent_dim, num_heads * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def _reshape_heads(self, x: Tensor, bsz: int, seq_len: int) -> Tensor:
        # [B, T, H*D_h] -> [B, H, T, D_h]
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_state: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        casual_mask: Tensor | None = None,
        pad_mask: Tensor | None = None,
        use_cache: bool = False,
    ):
        """
        参数：
            hidden_state: [B, T, hidden_dim]
            past_key_value: 可选，(past_k_latent, past_v_latent)，均为 [B, T_past, latent_dim]
            casual_mask: 可选，因果掩码 [T, T_total] 或 [1, 1, T, T_total]
            pad_mask: 可选，padding 掩码 [B, T_total]
            use_cache: 若为 True，返回 (output, new_past_key_value)

        返回：
            output 或 (output, new_past_key_value)
        """
        bsz, seq_len, _ = hidden_state.shape

        # Q: [B, T, H*D_h]
        q: Tensor = self.q_proj(hidden_state)
        # K/V（潜变量）: [B, T, D_latent]
        k_latent: Tensor = self.k_latent_proj(hidden_state)
        v_latent: Tensor = self.v_latent_proj(hidden_state)

        # 处理 KV cache（只拼接潜变量，节省内存）
        if past_key_value is not None:
            past_k_latent, past_v_latent = past_key_value
            k_latent = torch.cat([past_k_latent, k_latent], dim=1)  # [B, T_total, D_latent]
            v_latent = torch.cat([past_v_latent, v_latent], dim=1)

        new_past_key_value = (k_latent, v_latent) if use_cache else None

        # 将潜变量解码为各 head 的 K/V，并 reshape 到 [B, H, T_total, D_h]
        total_len = k_latent.shape[1]
        k_full = self.k_expand(k_latent)  # [B, T_total, H*D_h]
        v_full = self.v_expand(v_latent)  # [B, T_total, H*D_h]
        k = self._reshape_heads(k_full, bsz, total_len)
        v = self._reshape_heads(v_full, bsz, total_len)

        # Q 只用当前段长度（[B, H, T, D_h]）
        q = self._reshape_heads(q, bsz, seq_len)

        # 注意力分数：[B, H, T, T_total]
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if casual_mask is not None:
            # 支持 [T, T_total] 或已扩展形状；自动广播
            attn_score = attn_score + casual_mask * -1e9

        if pad_mask is not None:
            # [B, T_total] -> [B, 1, 1, T_total]
            pad_mask_exp = pad_mask.unsqueeze(1).unsqueeze(1)
            attn_score = attn_score + pad_mask_exp * -1e9

        attn_probs = F.softmax(attn_score, dim=-1)
        # 输出中间：[B, H, T, D_h]
        output_mid = torch.matmul(attn_probs, v)
        # 合并 head：[B, T, H*D_h]
        output_mid = output_mid.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output_mid)

        return (output, new_past_key_value) if use_cache else output


def mla_with_kv_cache_test():
    """使用 KV cache（缓存潜变量）按步前向，验证维度与缓存逻辑。"""
    bsz, seq_len, hidden_dim, num_heads, latent_dim = 2, 6, 32, 4, 16
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask_full = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # [T, T]

    mla = MultiHeadLatentAttention(hidden_dim, num_heads, latent_dim)
    past_key_value = None
    outs = []
    for t in range(seq_len):
        cur = hidden_state[:, t : t + 1, :]
        cur_mask = casual_mask_full[t : t + 1, : t + 1]  # [1, t+1]
        out_step, past_key_value = mla(
            cur,
            past_key_value=past_key_value,
            casual_mask=cur_mask,
            use_cache=True,
        )
        outs.append(out_step)
    out = torch.cat(outs, dim=1)
    print("mla step-by-step output shape:", out.shape)
    return out


def mla_full_seq_test():
    """整段前向，不使用缓存。"""
    bsz, seq_len, hidden_dim, num_heads, latent_dim = 2, 6, 32, 4, 16
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mla = MultiHeadLatentAttention(hidden_dim, num_heads, latent_dim)
    out = mla(hidden_state, casual_mask=casual_mask)
    print("mla full output shape:", out.shape)
    return out


if __name__ == "__main__":
    mla_full_seq_test()
    mla_with_kv_cache_test()


