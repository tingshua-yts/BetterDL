import torch
from torch.utils.data import SequentialSampler, BatchSampler
input = torch.tensor([1,2,3,4,5,6,7,8,9])
input_seq = SequentialSampler(input)
print(list(input_seq))
batch_data = BatchSampler(input_seq, batch_size = 3, drop_last=False)
print(list(batch_data))
