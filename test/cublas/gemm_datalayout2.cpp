#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

void printPlainMatrix(const float* matrix, const int H, const int W)
{
  for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
          std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, W)];
        }
        std::cout << std::endl;
    }
}

int main()
{
    cublasHandle_t handle;

    // Prepare input matrices
    float *A, *B, *C;
    int M, N, K;
    float alpha, beta;

    M = 2;
    N = 2;
    K = 3;
    alpha = 1.f;
    beta = 0.f;

    // create cuBLAS handle
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    cudaMallocManaged((void**)&A, sizeof(float) * M * K);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        A[IDX2C(i, j, K)] = i + 1;
      }
    }

    /*
      A:
      1.0000  1.0000  1.0000
      2.0000  2.0000  2.0000
    */
    std::cout << "A:" << std::endl;
    printPlainMatrix(A, M, K);


    cudaMallocManaged((void**)&B, sizeof(float) * K * N);
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < N; j++) {
        B[IDX2C(i, j, N)] = i + 1;
      }
    }
    /*
      B:
      1.0000  1.0000
      2.0000  2.0000
      3.0000  3.0000
    */
    std::cout << "B:" << std::endl;
    printPlainMatrix(B, K, N);

    cudaMallocManaged((void**)&C, sizeof(float) * M * N);
    for (int i = 0 ; i < M; i++) {
      for (int j = 0; j < N; j++) {
        C[IDX2C(i, j, N) ] = 1;
      }
    }

    std::cout << "C:" << std::endl;
    printPlainMatrix(C, M, N);

    // Gemm
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                B,
                N,
                A,
                K,
                &beta,
                C,
                N);

    cudaDeviceSynchronize();
    /*
      C out:
      6.0000 12.0000
      6.0000 12.0000
      C host memory layout:
      6.0000  12.0000 6.0000  12.0000
    */
    std::cout << "C out:" << std::endl;

    printPlainMatrix(C, M, N);

    std::cout << "C host memory layout:" << endl;;
    for (int i = 0; i < M *N ; ++i ) {
      cout << C[i] << "\t";
    }
    cout << endl;


    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
