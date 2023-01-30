DATA_DIR=/workspace/data/imageNet/ILSVRC/Data/CLS-LOC/
#DATA_DIR=/workspace/data/imageNet/ILSVRC/Data/CLS-LOC/
python opt/imagenet/test/train4imagenet.py  -a resnet50 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $DATA_DIR

#python opt/imagenet/test/train4imagenet.py -a resnet50 --gpu=0 $DATA_DIR