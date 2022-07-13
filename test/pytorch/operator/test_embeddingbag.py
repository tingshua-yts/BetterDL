import torch
from torch import nn
# embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
# print(embedding_sum)
# input=torch.tensor([1,2,4,5,4,3,2,9], dtype=torch.long)
# offsets = torch.tensor([0,4], dtype=torch.long)
# result = embedding_sum(input, offsets)
# print(result)

weight = torch.tensor([[0., 0., 0.,],
                       [1., 1., 1.,],
                       [2., 2., 2.,],
                       [3., 3., 3.,],
                       [4., 4., 4.,],
                       [5., 5., 5.,],
                       [6., 6., 6.,],
                       [7., 7., 7.,],
                       [8., 8., 8.,],
                       [9., 9., 9.,]])
input=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
offsets = torch.tensor([0,4], dtype=torch.long)
result = torch.nn.functional.embedding_bag(input, weight, torch.tensor([0]), mode='sum')
print("without offset")
print(result)
print("with offset")
result = torch.nn.functional.embedding_bag(input, weight, offsets, mode='sum')
print(result)
