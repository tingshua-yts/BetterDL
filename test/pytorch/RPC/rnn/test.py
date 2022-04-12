import torch.distributed.autograd as dist_autograd
with dist_autograd.context() as context_id:
    t1 = torch.rand((3, 3), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    loss = t1 + t2
    dist_autograd.backward(context_id, [loss.sum()])
    grads = dist_autograd.get_gradients(context_id)
    print(grads[t1])
    print(grads[t2])
