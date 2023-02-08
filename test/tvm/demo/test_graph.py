import torch
import torch.nn as nn
class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim

        self.weight = nn.Parameter(torch.randn(ndim, 1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x.mm(self.weight)  + self.bias

lm = LinearModel(5)
lm.eval()
image = torch.randn(4, 5)
input_names=["input0"]
output_names=["output0"]

torch.onnx.export(lm, image, './tmp/resnet50.onnx', verbose=True,input_names=input_names, output_names=output_names)