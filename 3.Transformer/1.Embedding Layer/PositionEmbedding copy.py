import torch 
from torch import nn, Tensor

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        pos = torch.arange(0, max_len, 1).unsqueeze(1).to(torch.float32)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-torch.log(torch.tensor(10000.0, dtype=torch.float32)) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x
def position_embedding_test():
    bsz, seq_len, hidden_dim, max_len = 6, 5, 10, 1000
    x = torch.randn(bsz, seq_len, hidden_dim)
    position_embedding = PositionEmbedding(max_len, hidden_dim)
    output = position_embedding(x)
    print(output.shape)
    return output
if __name__ == "__main__":
    position_embedding_test()
