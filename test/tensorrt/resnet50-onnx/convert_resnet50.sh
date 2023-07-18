set -e
set -x
trtexec --onnx=/mnt/model/resnet50-onnx/resnet50/model.onnx --saveEngine=/mnt/model/resnet50-onnx/resnet50/resnet_engine.trt
