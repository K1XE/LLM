import torch
from torch import nn, Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, X: Tensor): # [bsz, seq_len]
        embedded = self.embedding(X) # [bsz, seq_len, hidden_dim]
        return embedded

def token_embedding_test():
    bsz, seq_len, vocab_size, hidden_dim = 6, 5, 10, 16
    X = torch.randint(0, vocab_size, (bsz, seq_len))
    token_embedding = TokenEmbedding(vocab_size, hidden_dim)
    output = token_embedding(X)
    return output

if __name__ == "__main__":
    token_embedding_test()
