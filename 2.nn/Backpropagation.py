import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, Tensor
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, hidden_dim = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(2, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 3)

    def forward(self, x: Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def get_dummy_dataloader(bsz = 4):
    X = torch.randn(100, 2)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, bsz, shuffle=True)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(hidden_dim=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    dataloader = get_dummy_dataloader()
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        for bsz, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss: Tensor = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if bsz % 10 == 0:
                print(f"Eopch {epoch + 1}, Batch {bsz}, Loss: {loss.item(): .4f}")
if __name__ == "__main__":
    train() 