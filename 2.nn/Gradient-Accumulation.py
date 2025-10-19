import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, Tensor
import torch.nn.functional as F

class GradientAccumulation(nn.Module):
    def __init__(self, features, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(features, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 3)
    def forward(self, hidden_state: Tensor):
        hidden_state = self.linear1(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.linear2(hidden_state)
        return hidden_state
        

def get_dummy_dataloader(bsz_total = 1e3, features = 3, bsz = 4):
    X = torch.rand(bsz_total, features)
    y = torch.randint(0, 3, (bsz_total,))
    dataset = TensorDataset(X, y)
    ret = DataLoader(dataset, bsz)
    return ret

def train():
    gradient_accumulation_steps = 4
    bsz_total, features, mini_bsz, hidden_dim = 500, 10, 4, int(1e3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GradientAccumulation(features, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataloader = get_dummy_dataloader(bsz_total, features, mini_bsz)
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        for bsz, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if ((bsz + 1) % gradient_accumulation_steps == 0) or (bsz + 1 == len(dataloader)):
                print(f"Epoch [{epoch + 1} / {num_epochs}], Step [{bsz + 1}], Loss [{loss.item():.4f}]")
                optimizer.step()
                optimizer.zero_grad()
            
if __name__ == "__main__":
    train()
