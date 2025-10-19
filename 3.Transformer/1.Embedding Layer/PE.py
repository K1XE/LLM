import torch
from torch import nn, Tensor
from math import log
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, max_len, 2) * log(10000) / hidden_dim)
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        seq_len = x.shape[1]
        x += self.pe[:seq_len, :].unsqueeze(0)
        return x