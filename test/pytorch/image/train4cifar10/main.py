'''Train CIFAR10 with PyTorch.'''
from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from opt.cifar10.dataset.dali_cifar10_dataset import DaliCifarDataset
from opt.cifar10.dataloader.dali_cifar10_dataloader import DaliCifar10Dataloader
from opt.cifar10.transformer.dali_cifar10_transformer import DaliCifarTransformer

import os
import argparse
import time

from models import *
from utils import progress_bar, get_chrome_trace_handler

DATA_DIR="./tmp/data"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dali', action="store_true", help='whether use dali dataloader')
parser.add_argument("--batch_size","-bs", type=int, default="128")
parser.add_argument("--test_batch_size","-tbs", type=int, default="100")
parser.add_argument("--device_id", type=int, help="gpu device id", default=0)
parser.add_argument("--epoch", type=int, default=10)

args = parser.parse_args()
total_epoch = args.epoch
use_dali = args.dali
batch_size = args.batch_size
test_batch_size = args.test_batch_size
device_id = args.device_id
device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
checkpoint_path = "./checkpoint" if use_dali else "./checkpoint_pytorch"
# Data
print('==> Preparing data..')

data_start = time.time()
if use_dali:
    print("====> use dali dataloader")
    train_dali_transformer = DaliCifarTransformer(batch_size=batch_size, num_threads=8,
                                                  type="train", root="./tmp/data", device_id=device_id)
    trainloader = DaliCifar10Dataloader(train_dali_transformer, type="train")
    test_dali_transformer = DaliCifarTransformer(batch_size=test_batch_size, num_threads=8,
                                                 type="test", root="./tmp/data", device_id=device_id)
    testloader = DaliCifar10Dataloader(test_dali_transformer, type="test")
else:
    print("====> use pytorch dataloader")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=8)

prepare_time = time.time() - data_start
print(f'==> success Preparing data: {prepare_time}')

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
model_start = time.time()
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
print(f'==> success building model: {time.time() - model_start}')


if torch.cuda.is_available():
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch,prof):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if batch_idx == 0:
            print(f"input shape: {inputs.shape}, inputs device id: {inputs.get_device()}")
            print(f"target shape: {targets.shape}, targets device id: {targets.get_device()}")
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if prof:
            prof.step()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                print(f"input shape: {inputs.shape}")
                print(f"target shape: {targets.shape}")
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), test=True)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        torch.save(state, f'{checkpoint_path}/ckpt.pth')
        best_acc = acc

train_start = time.time()
# with torch.profiler.profile(
#         activities=[
#                 torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(skip_first=30, wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=get_chrome_trace_handler(prefix="tmp/trace/trace_resnet18_8_num_woker_"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
# ) as prof:
prof = None
for epoch in range(start_epoch,  start_epoch + total_epoch):
    ept = time.time()
    train(epoch, prof)
    test(epoch)
    scheduler.step()
    print(f"epoch {epoch} total time: {time.time() - ept}")
print(f"prepare data time: {prepare_time}, train total time: {time.time() - train_start}, total time: {time.time() - data_start}")