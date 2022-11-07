export MASTER_PORT=1234
export MASTER_ADDR=192.168.9.6
export WORLD_SIZE=2
export LOCAL_WORLD_SIZE=1
export RANK=1
export LOCAL_RANK=0
export LOGLEVEL="DEBUG"
python trian_multi_node.py
