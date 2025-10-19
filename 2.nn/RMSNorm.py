import torch
from torch import nn, Tensor

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps = 1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
    def forward(self, hidden_state: Tensor):
        rms = torch.sqrt(torch.mean(hidden_state.pow(2), dim=-1, keepdim=True) + self.eps)
        x_norm = hidden_state / rms
        output = self.gamma * x_norm
        return output
def rmsnorm_test():
    bsz, seq_len, hidden_dim = 8, 5, 16
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    rmsnorm = RMSNorm(hidden_dim, eps=1e-5)
    output = rmsnorm(hidden_state)
    return output
if __name__ == "__main__":
    rmsnorm_test()