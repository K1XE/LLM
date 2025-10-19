import torch
from torch import nn, Tensor
import torch.nn.functional as F
from MoE import SwiGLUExpert, MoEConfig, MoERouter

class SparseMoE(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.dropout = config.dropout
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(self.hidden_dim, self.dropout)
                for _ in range(self.num_experts)
            ]
        )
        self.router = MoERouter(self.hidden_dim, self.num_experts, self.top_k)

    def forward(self, X: Tensor):
        bsz, seq_len, hidden_dim = X.shape

        hidden_states = X.view(-1, hidden_dim)
        router_logits, router_weights, selected_experts_indices, expert_mask = (
            self.router(hidden_states)
        )

        ret_hidden_states = torch.zeros(
            (bsz * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # https://yuanchaofa.com/llms-zero-to-hero/the-way-of-moe-model-evolution.html#_2-%E7%89%88%E6%9C%AC2-sparsemoe-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%BD%BF%E7%94%A8:~:text=%23%20idx%20%E5%92%8C%20top_x%20%E9%83%BD%E6%98%AF%E4%B8%80%E7%BB%B4%20tensor
            # idx 和 top_x 都是一维 tensor
            # idx 的值是 0 或 1, 表示这个 token 是作为当前专家的 top1 还是 top2
            # top_x 的值是 token 在 batch*seq_len 中的位置索引
            # 例如对于 batch_size=2, seq_len=4 的输入:
            # top_x 的值范围是 0-7, 表示在展平后的 8 个 token 中的位置
            # idx 的值是 0/1, 表示这个 token 把当前专家作为其 top1/top2 专家

            cur_states = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)

            cur_hidden_states = expert_layer(cur_states) * router_weights[
                top_x, idx
            ].unsqueeze(-1)

            ret_hidden_states.index_add_(
                0, top_x, cur_hidden_states.to(hidden_states.dtype)
            )

        ret_hidden_states = ret_hidden_states.reshape(bsz, seq_len, hidden_dim)

        return ret_hidden_states, router_logits


def main():
    bsz, seq_len, hidden_dim = 5, 4, 10
    x = torch.rand(bsz, seq_len, hidden_dim)
    n_experts, top_k = 8, 3
    config = MoEConfig(hidden_dim, n_experts, top_k)
    token_level_moe = SparseMoE(config)
    out, router_logits = token_level_moe(x)
    print(out.shape)


if __name__ == "__main__":
    main()
