import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor data type in tvm
dtype = "float32"

# You will want to adjust the target to match any CPU vector extensions you
# might have. For example, if you're using using Intel AVX2 (Advanced Vector
# Extensions) ISA for SIMD, you can get the best performance by changing the
# following line to ``llvm -mcpu=core-avx2``, or specific type of CPU you use.
# Recall that you're using llvm, you can get this information from the command
# ``llc --version`` to get the CPU type, and you can check ``/proc/cpuinfo``
# for additional extensions that your processor might support.

target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))


################################
########## numpy baseline
# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

################################
########## naive te
# TVM Matrix Multiplication using TE

# 返回一个IterVar，用于遍历axis，这里返回的是一个range为0~1024的下标，用于A和B的计算
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
log = []

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="none", log=log)