import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("-nw", "--num_workers", type=int,  help="num workers for dataloader", default=2)
parser.add_argument("-bs", "--batch_size", type=int,  help="batch_size for dataloader", default=4)
parser.add_argument("-pf", "--prefetch_factor", type=int,  help="prefetch_factor for dataloader", default=2)


args = parser.parse_args()
num_worker = args.num_workers
batch_size = args.batch_size
prefetch_factor=args.prefetch_factor

print("start training with num_worker:" + str(num_worker))
t0 = time.time()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


trainset = torchvision.datasets.CIFAR10(root='./tmp/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_worker,
                                          prefetch_factor=prefetch_factor,
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./tmp/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_worker,
                                         prefetch_factor=prefetch_factor,
                                         pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device) ### net to device

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("tmp/trace/trace_gpu_numworker16_" + str(p.step_num) + ".json")

print("start training with num_worker:" + str(num_worker)  + " batch size:" + str(batch_size), " prefetch_factor:" + str(prefetch_factor))
# with profile(activities=[ProfilerActivity.CPU],profile_memory=True, record_shapes=True) as prof:
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training , total time {}'.format(time.time() - t0))