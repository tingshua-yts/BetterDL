import torch
from torch import nn
embedding = nn.Embedding(5, 3)
print("embedding table:")
for param in embedding.parameters():
    print(param)

print("embedding result:")
tensor = torch.LongTensor([4,3,2,1,0])
print(embedding(tensor))
