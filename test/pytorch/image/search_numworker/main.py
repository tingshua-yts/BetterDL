import torchvision
import torchvision.transforms as transforms
import torch

import numpy as np

from matplotlib import pyplot as plt

import os
import time
import resource
import threading
from resource import *

class MemoryMonitor(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.keep_measuring = True
    def run(self):
        self_max_usage = 0
        child_max_usage = 0
        while self.keep_measuring:
            child_cur_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            self_cur_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            self_max_usage = max(self_max_usage, self_cur_usage )
            child_max_usage = max(child_max_usage, child_cur_usage)
            print(f"self_cur_usage:{self_cur_usage}, self_max_usage:{self_max_usage}, child_cur_usage:{child_cur_usage}, child_max_usage:{child_max_usage}")
            time.sleep(2)
mm = MemoryMonitor()
mm.start()

if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    max_num_workers = os.cpu_count() #refer to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py

    plt.xlabel('num_workers')
    plt.ylabel('Total Time(Sec)')

    #batch_size_list = [ 2, 4, 8, 16, 32, 64, 128, 256]
    batch_size_list = [ 2,  8,  32, 64, 128]
    #num_workers_list = np.arange(max_num_workers + 1)
    num_workers_list = [2,  8, 16]
    for batch_size in batch_size_list:
        total_time_per_num_workers = []
        for num_workers in num_workers_list:
            loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                prefetch_factor=1)

            t1 = time.time()
            for _ in loader: pass
            t2 =time.time()

            total_time = t2 - t1
            total_time_per_num_workers.append(total_time)
            print(f"batch_size{batch_size}, num_workers{num_workers}, total_time(sec): ", total_time)

        plt.plot(num_workers_list, total_time_per_num_workers)
    plt.legend([f"batch size {batch_size}" for batch_size in batch_size_list])
    #plt.show()
    plt.savefig("num_worker.png")
