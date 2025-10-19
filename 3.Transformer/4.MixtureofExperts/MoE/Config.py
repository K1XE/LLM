import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MoEConfig:
    def __init__(self, hidden_dim, num_experts, top_k, dropout=0.1, shared_experts=2) -> None:
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        self.dropout = dropout
