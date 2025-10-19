import torch
from torch import nn, Tensor

class BatchNorm(nn.Module):
    def __init__(self, hidden_dim, eps = 1e-6, momentum = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.running_mean = torch.zeros(hidden_dim)
        self.running_var = torch.ones(hidden_dim)
    def forward(self, hidden_state: Tensor):
        if self.training:
            batch_mean = hidden_state.mean(dim=(0, 1), keepdim=False)
            batch_var = hidden_state.var(dim=(0, 1), keepdim=False, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            mean = self.running_mean
            var = self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (hidden_state - mean) / torch.sqrt(var + self.eps)
        output = self.gamma * x_norm + self.beta
        return output
def bn_test():
    bsz, seq_len, hidden_dim = 8, 5, 16
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    bn = BatchNorm(hidden_dim, eps=1e-5)
    output = bn(hidden_state)
    return output

if __name__ == "__main__":
    bn_test()