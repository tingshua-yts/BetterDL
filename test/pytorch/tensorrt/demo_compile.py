import torch
import torch_tensorrt
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# compile
model = ToyModel().eval().cuda()
trt_model = torch_tensorrt.compile(model,
    inputs= [torch_tensorrt.Input((2, 10))],
    #enabled_precisions= { torch_tensorrt.dtype.half} # Run with FP16
)

# infer
input_data = torch.rand(2, 10).to("cuda")
trt_result = trt_model(input_data)
with torch.no_grad():
    result = model(input_data)

torch.set_printoptions(precision=20)
print(result)
print(trt_result)
print(torch.equal(trt_result, result))

# tensor([[False, False,  True,  True,  True],
#         [ True, False, False, False,  True]], device='cuda:0')
print(torch.eq(trt_result, result))



# save
torch.jit.save(trt_model, "trt_ts_module.ts")
