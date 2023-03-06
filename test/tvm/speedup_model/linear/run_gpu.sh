set -e
set -x
g++ cpp_deploy_gpu.cpp -o run_linear_gpu -std=c++17 -O2 -fPIC -g \
    -I /mnt/tvm/apache-tvm-src-v0.10.0/include \
    -I /mnt/tvm/apache-tvm-src-v0.10.0/3rdparty/dmlc-core/include \
    -I /mnt/tvm/apache-tvm-src-v0.10.0/3rdparty/dlpack/include \
    -L /mnt/tvm/apache-tvm-src-v0.10.0/build/ -ldl -pthread -ltvm_runtime

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/mnt/tvm/apache-tvm-src-v0.10.0/build/
./run_linear_gpu