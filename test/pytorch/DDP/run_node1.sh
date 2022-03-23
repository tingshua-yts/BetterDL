# python -m torch.distributed.launch --nproc_per_node=4 \
#        --nnodes=2 --node_rank=1 --master_addr="192.168.9.104" \
#        --master_port=1234 trian_multi_node.py
torchrun --nproc_per_node=4 \
	 --nnodes=2 \
	 --node_rank=1\
	 --master_addr="192.168.9.104" \
	 --master_port=1234\
	 trian_multi_node.py
