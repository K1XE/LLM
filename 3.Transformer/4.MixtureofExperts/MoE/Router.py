import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, X: Tensor):
        # [bsz * seq, num_experts]
        router_logits = self.gate(X)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)

        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True).to(
            X.dtype
        )
        # [n_experts, top_k, bsz * seq]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)

        return router_logits, router_weights, selected_experts, expert_mask
