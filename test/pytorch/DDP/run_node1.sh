# python -m torch.distributed.launch --nproc_per_node=4 \
#        --nnodes=2 --node_rank=1 --master_addr="192.168.9.104" \
#        --master_port=1234 trian_multi_node.py
# export LOGLEVEL="DEBUG"
# torchrun --nproc_per_node=4 \
# 	 --nnodes=2 \
# 	 --node_rank=1\
# 	 --master_addr="192.168.9.6" \
# 	 --master_port=1234\
# 	 trian_multi_node.py

# run cluster info with env ags
# export PET_MASTER_ADDR=192.168.9.6
# export PET_MASTER_PORT=1234
# export PET_NPROC_PER_NODE=4
# export PET_NNODES=2
# export PET_NODE_RANK=1
# export LOGLEVEL="DEBUG"
# torchrun trian_multi_node.py
