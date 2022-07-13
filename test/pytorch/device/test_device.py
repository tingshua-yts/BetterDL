import torch
import torch.nn

torch.cuda.set_device(1)
x = torch.LongTensor([1,2,3,4])
print(x)
x = x.cuda()
print(x)

y = torch.LongTensor([1,2,3,4]).cuda()
z = x + y

x1 = torch.LongTensor([1,2,3,4])
y1 = torch.LongTensor([1,2,3,4])
z1 = x1 + y1
print(z1)
