import torch
from torch.utils.data import SequentialSampler, BatchSampler,RandomSampler
#input = torch.tensor([1,2,3,4,5,6,7,8,9])
input = ['a','b','c','d','e','f','g']
input_seq = SequentialSampler(input)
print(list(input_seq))
print(list(torch.utils.data.DataLoader(input, sampler=input_seq, num_workers=0)))

rand_data = RandomSampler(input)
print(list(rand_data))
print(list(torch.utils.data.DataLoader(input, sampler=rand_data, num_workers=0)))

batch_data = BatchSampler(input_seq, batch_size = 3, drop_last=False)
print(list(batch_data))





