import torch
import torch.nn as nn

m = nn.Softmax()
input = torch.tensor([[.1,2.,3.],[4.,5.,6.]], requires_grad=True)
softmax_output = m(input)
print("softmax output:")
print(softmax_output.data)

target = torch.tensor([1, 2])
loss = nn.NLLLoss()
loss_out = loss(softmax_output, target)
print("lass output:")
print(loss_out.data)
