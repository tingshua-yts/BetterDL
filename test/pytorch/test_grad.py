import torch
X = torch.tensor([2., 4.], requires_grad=True)
Y = X.sum().pow(2)
Y.backward()
print(X.grad)
