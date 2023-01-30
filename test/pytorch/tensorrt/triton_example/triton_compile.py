import torch
import torch_tensorrt
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to("cuda")

# Compile with Torch TensorRT;
trt_model = torch_tensorrt.compile(model,
    inputs= [torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions= { torch.half} # Run with FP32
)

# Save the model
torch.jit.save(trt_model, "./tmp/model/resnet50/1/model.pt")