tvmc run \
--inputs ./tmp/imagenet_cat.npz \
--output ./tmp/predictions.npz \
--print-time \
--repeat 100 \
./tmp/resnet50-v2-7-tvm_autotuned.tar