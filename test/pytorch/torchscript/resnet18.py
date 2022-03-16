import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
dummy_input = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, dummy_input)

# Save the TorchScript model
traced_script_module.save("resnet18.pt")
