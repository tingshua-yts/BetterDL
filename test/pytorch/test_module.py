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
x = torch.randn(4, 5)
print(lm(x))
