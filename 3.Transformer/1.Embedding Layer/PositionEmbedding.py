import torch
import math
from torch import nn, Tensor

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        position = torch.arange(0, max_len, 1).unsqueeze(1).float() # [max_len, 1]
        # [hidden_dim / 2]
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim) 
        # [max_len, 1] * [hidden_dim / 2] -> [max_len, 1] * [1, hidden_dim / 2]
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, X: Tensor):
        _, seq_len, _ = X.shape
        X = X + self.pe[:seq_len, :].unsqueeze(0) # [bsz, seq_len, hidden_dim] + [1, seq_len, hidden_dim]
        return X
    
def position_embedding_test():
    bsz, seq_len, hidden_dim, max_len = 6, 4, 10, 1000
    x = torch.randn(bsz, seq_len, hidden_dim)
    position_embedding = PositionEmbedding(max_len, hidden_dim)
    output = position_embedding(x)
    print(output.shape)
    return output

if __name__ == "__main__":
    position_embedding_test()