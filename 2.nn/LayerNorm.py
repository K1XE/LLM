import torch
from torch import nn, Tensor
class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps = 1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, hidden_state: Tensor):
        mean = hidden_state.mean(dim=-1, keepdim=True)
        variance = hidden_state.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (hidden_state - mean) / torch.sqrt(variance + self.eps)
        ouput = self.gamma * x_norm + self.beta
        return ouput
def ln_test():
    bsz, seq_len, hidden_dim = 8, 5, 16
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    ln = LayerNorm(hidden_dim, eps=1e-5)
    output = ln(hidden_state)
    return output

if __name__ == "__main__":
    ln_test()