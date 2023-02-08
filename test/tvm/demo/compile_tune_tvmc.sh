tvmc compile \
--target "llvm" \
--tuning-records ./tmp/resnet50-v2-7-autotuner_records.json  \
--output ./tmp/resnet50-v2-7-tvm_autotuned.tar \
./tmp/resnet50-new.onnx