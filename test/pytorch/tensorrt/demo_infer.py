import torch
import torch_tensorrt

trt_ts_module = torch.jit.load("trt_ts_module.ts")
input_data = torch.rand(2, 10).to("cuda")
result = trt_ts_module(input_data)
print(result)
torch.set_printoptions(precision=20)
print(torch.tensor([1.1234567]))