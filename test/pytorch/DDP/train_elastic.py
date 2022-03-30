import argparse
import os
import sys
import time
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimize_state_dict": optimizer.state_dict(),
}, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    model = ToyModel().cuda(local_rank)
    ddp_model = DDP(model, [local_rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    max_epoch = 100
    first_epoch = 0
    ckp_path = "checkpoint.pt"
    if os.path.exists(ckp_path):
        print(f"load checkpoint from {ckp_path}")
        checkpoint = load_checkpoint(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimize_state_dict"])
        first_epoch = checkpoint["epoch"]

    for i in range(first_epoch, max_epoch):
        time.sleep(1)
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 5).to(local_rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        print(f"[{os.getpid()}] epoch {i} (rank = {rank}, local_rank = {local_rank}) loss = {loss.item()}\n")
        optimizer.step()
        save_checkpoint(i, model, optimizer, ckp_path)

def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
