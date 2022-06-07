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

def hook(*ignore):
    print("aaa")

for k, v in lm.named_parameters():
    print(k)
    print(v)
# for p in lm.parameters():
#     p.grad = p.data.new(p.size()).zero_()
#     p_tmp = 3 * p
#     print(p)
#     print(p_tmp)
#     print(p_tmp.grad_fn)
#     print(p_tmp.grad_fn.next_functions[0][0])



# x = torch.randn(4, 5)
# p = lm(x)
# l = p.sum()
# l.backward()
# #print(l.grad_fn)

# print("\n---------------")
# print(l.grad_fn.next_functions)
# print(l.grad_fn.next_functions[0][0])
# print("\n---------------")
# print(l.grad_fn.next_functions[0][0].next_functions)
# print(l.grad_fn.next_functions[0][0].next_functions[0][0])
# print(l.grad_fn.next_functions[0][0].next_functions[1][0])

# print("\n---------------")
# print(l.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)

# print("\n---------------")
# print(l.grad_fn.next_functions[0][0].next_functions[1][0].next_functions)
