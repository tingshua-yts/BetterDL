import torch
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
#print(resnet50)
image = torch.randn(1, 3, 244, 244)
input_names=["data"]
output_names=["output0"]

torch.onnx.export(resnet50, image, './tmp/resnet50-new.onnx', verbose=True,input_names=input_names, output_names=output_names)