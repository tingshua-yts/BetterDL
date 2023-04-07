import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import logging
logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)
from torch.nn.parallel import DistributedDataParallel as DDP


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG'] = "INFO"
    os.environ['WORLD_SIZE'] = '3'
    print(str(os.getpid()) + ":" + str(rank) + ":" + world_size)
    # create default process group
    dist.init_process_group(backend="nccl", rank=rank)
    print(f"world size after init: {torch.distributed.get_world_size()}")
    print(f"is dist inited:{torch.distributed.is_initialized()}")

    # create local model
    model = nn.Linear(10, 10).to(rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)

    # backward pass
    loss_fn(outputs, labels).backward()

    # update parameters
    optimizer.step()

def main():
    worker_size = 3
    mp.spawn(run_worker,
        args=("hey",),
        nprocs=worker_size,
        join=True)

if __name__=="__main__":
    main()
