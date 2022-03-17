#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
print(">>>>>>> dataset info <<<<<<<<<<<<")
print("dataset len:" + str(len(dataset)))
print("dataset elem type: " + str(type(dataset[0])))
print("dataset elem len: " + str(len(dataset[0])))
# print("dataset elem index 1 type: " + str(type(dataset[0][0])))
# print("dataset elem index 2 type: " + str(type(dataset[0][1])))


# print(">>>>>>> dataloader info <<<<<<<<<<<<")
# train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
