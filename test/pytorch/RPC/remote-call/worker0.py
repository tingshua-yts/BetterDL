# On worker 0:
import os
import torch
import torch.distributed.rpc as rpc
import time
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

rpc.init_rpc("worker0", rank=0, world_size=2)
for i in range(10):
    time.sleep(1)
    print(u"sleep {i}")
ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
print(type(ret))
print(ret)
rpc.shutdown()
