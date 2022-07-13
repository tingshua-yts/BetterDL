import torch
import torch.nn as nn
tensor = torch.rand(4)
print("origin:")
print(tensor)
m = nn.Dropout(p=0.5)
print("dropout result:")
print(m(tensor))

tensor2 = torch.rand(2,3)
print("origin:")
print(tensor2)
print("dropout result:")
print(m(tensor2))
