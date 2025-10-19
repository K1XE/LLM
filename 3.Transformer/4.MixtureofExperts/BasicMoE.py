import torch
from torch import nn, Tensor
from MoE import SwiGLUExpert

class BasicMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_dim, 0.05) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, X: Tensor):
        # [bsz, n_experts]
        expert_weights: Tensor = self.gate(X)
        expert_output_list: Tensor = [expert(X).unsqueeze(1) for expert in self.experts]
        # [bsz, n_experts, hidden_dim]
        expert_output = torch.cat(expert_output_list, dim=1)
        # [bsz, 1, n_experts]
        expert_weights = expert_weights.unsqueeze(1)
        # [bsz, 1, hidden_dim]
        out: Tensor = expert_weights @ expert_output
        return out.squeeze(dim=1)
