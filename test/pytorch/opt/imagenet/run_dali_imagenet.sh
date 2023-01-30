#DATA_DIR=/workspace/data/imageNet/ILSVRC/Data/CLS-LOC-MINI/
DATA_DIR=/workspace/data/imageNet/ILSVRC/Data/CLS-LOC/
python opt/imagenet/test/train4imagenet.py --dali -a resnet50 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $DATA_DIR

