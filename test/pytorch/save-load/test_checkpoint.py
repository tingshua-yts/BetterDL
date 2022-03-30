import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

PATH="model.pt"
torch.save({
    "epoch": 10,
    "loss": 0.001,
    "model_state_dict": model.state_dict(),
    "optimize_state_dict": optimizer.state_dict(),
}, PATH)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimize_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
print(f"checkpoint epoch:{epoch}")
print(f"checkpint loss:{loss}")
