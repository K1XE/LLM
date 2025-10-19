import torch
from torch import nn, Tensor

class Dropout(nn.Module):
    def __init__(self, hidden_dim, dropout_rate = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

    def forward(self, hidden_state: Tensor):
        if self.training:
            mask = (torch.rand(hidden_state.shape) > self.dropout_rate).float()
            output = mask * hidden_state / (1 - self.dropout_rate)
        else:
            output = hidden_state
        return output
def dropout_test():
    bsz, seq_len, hidden_dim = 2, 4, 8
    hidden_state = torch.randn(bsz, seq_len, hidden_dim)
    dropout = Dropout(hidden_dim, dropout_rate=0.15)
    dropout.train()
    output_train = dropout(hidden_state)

    dropout.eval()
    output_eval = dropout(hidden_state)
    
if __name__ == "__main__":
    dropout_test()