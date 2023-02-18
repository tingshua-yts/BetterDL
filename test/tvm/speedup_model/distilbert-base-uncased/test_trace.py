import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

    def weighted_kernel_sum(self, weight):
        return weight * self.conv.weight


n = Net()
example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 1, 3, 3)

# Trace a specific method and construct `ScriptModule` with
# a single `forward` method
module = torch.jit.trace(n.forward, example_forward_input)

# Trace a module (implicitly traces `forward`) and construct a
# `ScriptModule` with a single `forward` method
module = torch.jit.trace(n, example_forward_input)

# Trace specific methods on a module (specified in `inputs`), constructs
# a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
module = torch.jit.trace_module(n, inputs)