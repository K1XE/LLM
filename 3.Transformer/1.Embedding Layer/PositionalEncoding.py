from torch import nn, Tensor
import torch
import torch.nn.functional as F
import math
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(- torch.arange(0, hidden_dim, 2) / hidden_dim * math.log(10000))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)
    def forward(self, X):
        _, seq_len, _ = X.shape
        X += self.pe[:seq_len, :].unsqueeze(0)
        return X
    
def main():
    bsz, seq_len, hidden_dim = 5, 3, 4
    max_len = 10
    X = torch.randn(bsz, seq_len, hidden_dim)
    PE = PositionalEncoding(max_len, hidden_dim)
    output = PE(X)
    print(output.shape)
    return output

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    main()