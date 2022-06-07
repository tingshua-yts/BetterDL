import torch
X = torch.tensor([2., 4.], requires_grad=True)
Y = X.sum().pow(2)
print(Y)
dY = torch.tensor(1.)
# Y.backward()
# print("without hook:")
# print(X.grad)


########## test register ##########
# X1 = torch.tensor([2., 4.], requires_grad=True)
# Y1 = X1.sum().pow(2)
# print("with  hook:")
# Y1.register_hook(lambda grad: grad * 2)
# Y1.backward()
# print(X1.grad)
