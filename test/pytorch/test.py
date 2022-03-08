import torch
# size 2 * 3
x = torch.tensor([[1, 1, 1],
                  [1, 1, 1]])

# size 3 * 2
w = torch.tensor([[2, 2],
                  [2, 2],
                  [2, 2]])

r = torch.mm(x, w)
print(r)
# bias 1

b = torch.tensor([1])

print(r + b)
