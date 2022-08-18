import time
import sys
sys.path.append("./")
from opt.cifar10.dataset.dali_cifar10_dataset import DaliCifarDataset
from opt.cifar10.dataloader.dali_cifar10_dataloader import DaliCifar10Dataloader
from opt.cifar10.transformer.dali_cifar10_transformer import DaliCifarTransformer
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

############ global variable & func ############
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR_IMAGES_NUM_TRAIN = 50000
CIFAR_IMAGES_NUM_TEST = 10000
IMG_DIR = './tmp/data/'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
CROP_SIZE = 32

def test(data_loader, data_loader_type):
    for i, data in enumerate(data_loader):
        # 这里image已经在gpu上了，其实不需要转换，labels是需要转换的
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
        if i == 0:
            print(f"image device_id: {data[0].get_device()}")
            print(f"labels device_id: {data[1].get_device()}")

    print(f'[{data_loader_type}] end train dataloader iteration')

    print(f"[{data_loader_type}] test dataloader length: %d"%len(data_loader))
    print(f'[{data_loader_type}] start iterate test dataloader')

############# dali dataloader ################
t1 = time.time()
dali_transformer = DaliCifarTransformer(batch_size=TRAIN_BS, num_threads=8, type="train", root="./tmp/data", device_id=0)
dali_dataloader = DaliCifar10Dataloader(dali_transformer, type="train")
t2 = time.time()
test(dali_dataloader, "DALI")
t3 = time.time()
print(f"[DALI] train time total: {t3 - t1}, init: {t2 - t1}, iterate: {t3 - t2}")

############# pytorch dataloader ################
t1 = time.time()
transform_train = transforms.Compose([
    transforms.RandomCrop(CROP_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
train_dst = CIFAR10(root=IMG_DIR, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dst, batch_size=TRAIN_BS, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
t2 = time.time()
print("[PyTorch] train dataloader length: %d"%len(train_loader))
print('[PyTorch] start iterate train dataloader')
test(train_loader, "pytorch")
t3 = time.time()
print(f"[Pytorch] train time total: {t3 - t1}, init: {t2 - t1}, iterate: {t3 - t2}")

"""result: DALI init花费时间较长，但是可以压缩iterate时间
image device_id: 0
labels device_id: -1
[DALI] end train dataloader iteration
[DALI] test dataloader length: 196
[DALI] start iterate test dataloader
[DALI] train time total: 34.11853265762329, init: 33.13549017906189, iterate: 0.9830424785614014

Files already downloaded and verified
[PyTorch] train dataloader length: 196
[PyTorch] start iterate train dataloader
image device_id: -1
labels device_id: -1
[pytorch] end train dataloader iteration
[pytorch] test dataloader length: 196
[pytorch] start iterate test dataloader
[Pytorch] train time total: 5.37505030632019, init: 1.01499342918396, iterate: 4.3600568771362305
"""