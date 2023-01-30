torchrun \
    --nnodes=1:3\
    --nproc_per_node=4\
    --max_restarts=3\
    --rdzv_id=1\
    --rdzv_backend=c10d\
    --rdzv_endpoint="192.168.9.18:1234"\
    imagenet.py --data='/mnt/data/imageNet/ILSVRC/Data/CLS-LOC-MINI'