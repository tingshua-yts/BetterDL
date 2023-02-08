import tvm
import tvm.testing
from tvm import te
import numpy as np
from eval import *
from tvm.contrib import cc
from tvm.contrib import utils


# create computation
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

# create and config schedule
s = te.create_schedule(C.op)

################################################################################
# Finally we must bind the iteration axis bx and tx to threads in the GPU
# compute grid. The naive schedule is not valid for GPUs, and these are
# specific constructs that allow us to generate code that runs on a GPU.
bx, tx = s[C].split(C.op.axis[0], factor=64)
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

# build
tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

# run

dev = tvm.device(tgt_gpu.kind.name, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
log=[]
evaluate_addition(fadd, tgt_gpu, "gpu", log, A, B, C)

# save and export
temp = utils.tempdir()
fadd.save(temp.relpath("myadd.o"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
fadd.imported_modules[0].save(temp.relpath("myadd.cubin"))
print("save and export success")
print(temp.listdir())

# load
print("load module")
fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cubin"))
fadd1.import_module(fadd1_dev)

# assert
fadd1(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
print("run success")