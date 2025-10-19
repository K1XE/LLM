import torch
from torch import nn, Tensor
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, hidden_size)
    def forward(self, X):
        output = self.emb(X)
        return output