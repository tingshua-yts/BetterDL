#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
/**
 * Minimal example to test add tensor
 * using cuDNN.
 **/
int main(int argc, char** argv)
{
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  std::cout << "Found" << numGPUs << " GPUs." << std::endl;
  cudaSetDevice(1);

  int device;
  struct cudaDeviceProp devProp;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&devProp, device);
  std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

  // create cudnnhandle
  cudnnHandle_t handle_;
  cudnnCreate(&handle_);
  std::cout << "Create cuDNN handle" << std::endl;

  // create A tensor descriptor
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  int n = 1, c = 1, h = 4, w = 4;
  int NUM_ELEMENTS = n*c*h*w;
  cudnnTensorDescriptor_t a_desc;
  cudnnCreateTensorDescriptor(&a_desc);
  cudnnSetTensor4dDescriptor(a_desc, format, dtype, n, c, h, w);

  // create tensor memory
  float *a;
  cudaMallocManaged(&a, NUM_ELEMENTS * sizeof(float));
  for(int i =0; i < NUM_ELEMENTS; i++) {

    a[i] = i * 1.00f;
  }

  float *b;
  cudaMallocManaged(&b, NUM_ELEMENTS * sizeof(float));
  for(int i =0; i < NUM_ELEMENTS; i++) {

    b[i] = i * 1.00f;
  }


  // compute add
  float alpha[1] = {1 };
  float beta[1] = {
    2
  };

  int ret = cudnnAddTensor(
      handle_,
      alpha,
      a_desc,
      a,
      beta,
      a_desc,
      b);
  if (ret != CUDNN_STATUS_SUCCESS) {
    std::cout << "cudnn add fail" << ret << std::endl;
  }

  cudnnDestroy(handle_);
  std::cout << std::endl << "Destroyed cuDNN handle." << std::endl;
  std::cout << "New array: ";
  for(int i=0;i<NUM_ELEMENTS;i++) std::cout << b[i] << " ";
  std::cout << std::endl;
  cudaFree(a);
  cudaFree(b);







}
