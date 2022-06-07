import torch
x = torch.tensor([3., 3., 3., 3.], requires_grad=True)
y = torch.tensor([2., 2., 2., 2.], requires_grad=True)
z = x * y
l = z.sum()

print(x)
print(y)
print(z) # no grad, 只有叶子节点才有
print(l) # no grad

# print("------ backward reult --------")
# l.backward()
# print(x.grad)
# print(y.grad)
# backward

print("------ manul backward reult of grad_fn --------")
# dl = torch.tensor(1.) # TODO: 为什么是1
# print(dl)
# back_sum = l.grad_fn
# dz = back_sum(dl)
# print(dz)
# back_mul = z.grad_fn
# dx, dy = back_mul(dz)
# print(dx)
# print(dy) # 到这里为止已经计算出来了x和y的梯度，但是并没有更新到tensor的grad字段

print("------ manul backward reult of next --------")
dl = torch.tensor(1.) # TODO: 为什么是1
print("dl:")
print(dl) # tensor(1.)
back_sum = l.grad_fn
dz = back_sum(dl)

back_mul = back_sum.next_functions[0][0]
print("back_sum.next_functions:")
print(back_sum.next_functions) # ((<MulBackward0 object at 0x7f2e74b21520>, 0),)

dx, dy = back_mul(dz)
print(dx) # tensor([2., 2., 2., 2.], grad_fn=<MulBackward0>)
print(dy) # tensor([3., 3., 3., 3.], grad_fn=<MulBackward0>)
print("back_mul.next_functions")
print(back_mul.next_functions) # ((<AccumulateGrad object at 0x7f2e73c2e040>, 0), (<AccumulateGrad object at 0x7f2e73c2e100>, 0))
back_x = back_mul.next_functions[0][0] # AccumulateGrad 处理叶子节点grad的计算
back_x(dx)
back_y = back_mul.next_functions[1][0]
back_y(dy)
print(x.grad)
print(y.grad)
