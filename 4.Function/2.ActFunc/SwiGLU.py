from torch import nn, Tensor
import torch

class LlamaMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim)
    def forward(self, x: Tensor):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        gate_swish = nn.SiLU(gate)
        gate_swish = gate * torch.sigmoid(gate)
        output = self.down_proj(up * gate_swish)
        return output
    
def swiglu_test():
    x = torch.randn(6, 5, 10)
    SwiGLU = LlamaMLP(10, 20)
    output = SwiGLU(x)
    print(output)
    return output

if __name__ == "__main__":
    swiglu_test()