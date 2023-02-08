import tvm
import tvm.testing
from tvm import te
import numpy as np
from eval import *

# create commputation
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

# create schedule
s = te.create_schedule(C.op)

# build
tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512", host="llvm")
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

# run
dev = tvm.device(tgt.kind.name, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


log = []
evaluate_addition(fadd, tgt, "naive", log, A, B, C)


# parallel opt
s[C].parallel(C.op.axis[0])
fadd_parallel = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
fadd_parallel(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluate_addition(fadd_parallel, tgt, "parallel", log, A, B, C)

