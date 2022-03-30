export MASTER_ADDR="192.168.9.104"
export MASTER_PORT="1234"
#export LOGLEVEL="DEBUG"
torchrun \
    --nnodes=1:3\
    --nproc_per_node=4\
    --max_restarts=3\
    --rdzv_id=1\
    --rdzv_backend=c10d\
    --rdzv_endpoint="192.168.9.104:1234"\
     train_elastic.py
