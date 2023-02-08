# This may take several minutes depending on your machine
tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,244,244]" \
--output "tmp/resnet50-new-tvm.tar" \
"./tmp/resnet50-new.onnx"