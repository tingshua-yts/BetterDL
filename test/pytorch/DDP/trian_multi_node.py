import argparse
import os
import sys
import tempfile
from time import sleep
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datetime


from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    while True:
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
        model = ToyModel().cuda(local_rank)
        ddp_model = DDP(model, [local_rank])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 5).to(local_rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) loss = {loss.item()}\n")
        optimizer.step()
        sleep(1)

def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=30))
    #dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
