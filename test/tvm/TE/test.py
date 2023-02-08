import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

# declare some variables for use later
n = te.var("n")
m = te.var("m")

A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))
