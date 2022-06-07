# On worker 1:
import os
import torch.distributed.rpc as rpc
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
rpc.init_rpc("worker1", rank=1, world_size=2)
rpc.shutdown(graceful=False)
print("shutdown complete")
