import torch
import torch.profiler
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=1000, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))

def train(model):
    num_classes = 1000
    num_batches = 5
    batch_size = 120
    image_w = 128
    image_h = 128

    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./result'),
            record_shapes=True,
            profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
            with_stack=True
    ) as p:
        for i in range(num_batches):
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                          .scatter_(1, one_hot_indices, 1)

            # run forward pass
            optimizer.zero_grad()
            outputs = model(inputs.to('cuda:0'))

            # run backward pass
            labels = labels.to(outputs.device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            print("epoch: " + str(i) + ", loss: " + str(loss.item()))
            optimizer.step()
            p.step()  # 不要忘记对profile manager进行迭代
if __name__=="__main__":
    model = ModelParallelResNet50()
    train(model)
