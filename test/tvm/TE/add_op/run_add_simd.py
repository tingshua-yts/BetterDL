import tvm
import tvm.testing
from tvm import te
import numpy as np
from eval import *

# Recreate the schedule, since we modified it with the parallel operation in
# the previous example
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)


# This factor should be chosen to match the number of threads appropriate for
# your CPU. This will vary depending on architecture, but a good rule is
# setting this factor to equal the number of available CPU cores.
factor = 4

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512", host="llvm")
fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

log = []

evaluate_addition(fadd_vector, tgt, "vector", log, A, B, C)

#print(tvm.lower(s, [A, B, C], simple_mode=True))