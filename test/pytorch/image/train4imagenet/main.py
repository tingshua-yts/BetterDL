import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torch

trainset = torchvision.datasets.ImageNet(root='./tmp/data', train=True, download=True)

testset = torchvision.datasets.ImageNet(root='./tmp/data', train=False, download=True)
