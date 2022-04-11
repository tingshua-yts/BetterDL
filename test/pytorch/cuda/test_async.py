import torch
from torch import nn
import torch.profiler
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck

batch_size = 120
image_w = 128
image_h = 128
num_classes = 1000

class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)


# model0 =models.resnet152(pretrained=True).to("cuda:0")
# model1 = models.resnet152(pretrained=True).to("cuda:1")
# model2 = models.resnet152(pretrained=True).to("cuda:2")


model0 = ResNet50().to("cuda:0")
model1 = ResNet50().to("cuda:1")
model2 = ResNet50().to("cuda:2")

with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./resnet50'),
            record_shapes=True,
            profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
            with_stack=True) as p:
    # embedding = nn.Embedding(512, 256)
    # embedding = embedding.to("cuda:0")
    # tensor = torch.randint(0, 255,(128,)).to("cuda:0")
    for i in range(8):
        print("epoch: " + str(i))
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        outputs0 = model0(inputs.to('cuda:0'))
        outputs1 = model1(inputs.to('cuda:1'))
        outputs2 = model2(inputs.to('cuda:2'))

        p.step()
