

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <torch/extension.h>
using data_type = float;
using namespace std;

#define index(i, j, ld) ((i) * (ld) + (j))

// cuda API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
        }                                                                                          \
    } 


__global__ void showGPU(data_type *ptr, int size){
    for (int i = 0; i < size; i++)
        printf("%-3.1f ", ptr[i]);
    printf("\n");
    
}

void printTensor(at::Tensor tensor) {
    auto sizes = tensor.sizes();
    auto data_accessor = tensor.accessor<float, 2>(); // Assuming tensor is 2D
    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            printf("%f ", data_accessor[i][j]);
        }
        printf("\n");
    }
}
void printMatrix(const std::vector<at::Tensor>& vectorOfTensors) {
    for (size_t i = 0; i < vectorOfTensors.size(); i++) {
        printf("Tensor %zu:\n", i);
        printTensor(vectorOfTensors[i]);
        printf("\n");
    }
}

std::vector<at::Tensor>& multi_group(
                const std::vector<at::Tensor>& A,
                const std::vector<at::Tensor>& B,
                std::vector<at::Tensor>& C){
   //断言
   int batchCount = A.size();
   int* m = new int[batchCount];
   int* n = new int[batchCount];
   int* k = new int[batchCount];
   data_type* alpha = new data_type[batchCount]; 
   data_type* beta = new data_type[batchCount]; 
   int* lda = new int[batchCount];
   int* ldb = new int[batchCount];
   int* ldc = new int[batchCount];
   int* group_size = new int[batchCount];
   cublasOperation_t* transa_array = new cublasOperation_t[batchCount];
   cublasOperation_t* transb_array = new cublasOperation_t[batchCount];
   
   for(int i = 0; i < batchCount; i++){
       m[i] = A[i].sizes()[0];
       n[i] = B[i].sizes()[1];
       k[i] = A[i].sizes()[1]; 
       alpha[i] = 1.0;
       beta[i] = 0.0;
       lda[i] = m[i];
       ldb[i] = k[i];
       ldc[i] = m[i]; 
       group_size[i] = 1;
       transa_array[i] = CUBLAS_OP_N;
       transb_array[i] = CUBLAS_OP_N;
   }
   
   // const int m = A[0].sizes()[0];
   // const int n = B[0].sizes()[1];
   // const int k = A[0].sizes()[1];
   // int batchCount = A.size();
    // 打印tensor数据
  //  printf("A\n");
    //printMatrix(A);
  //  printf("B\n");
    //printMatrix(B);

   
    std::vector<data_type*> d_A(batchCount, nullptr);
    std::vector<data_type*> d_B(batchCount, nullptr);
    std::vector<data_type*> d_C(batchCount, nullptr);

    data_type **d_A_array = nullptr, **d_B_array = nullptr, **d_C_array = nullptr;

    // 传输数据
    for (int i = 0; i < batchCount; i++) {
        d_A[i] = A[i].contiguous().data_ptr<data_type>();
        d_B[i] = B[i].contiguous().data_ptr<data_type>();
        d_C[i] = C[i].contiguous().data_ptr<data_type>();
        }
  //  printf("test00\n");
    // 分配 array
    CUDA_CHECK(cudaMalloc(&d_A_array, sizeof(data_type*) * batchCount));
    CUDA_CHECK(cudaMalloc(&d_B_array, sizeof(data_type*) * batchCount));
    CUDA_CHECK(cudaMalloc(&d_C_array, sizeof(data_type*) * batchCount));

    // 传输数据
    CUDA_CHECK(cudaMemcpy(d_A_array, d_A.data(), sizeof(data_type*) * batchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_array, d_B.data(), sizeof(data_type*) * batchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_array, d_C.data(), sizeof(data_type*) * batchCount, cudaMemcpyHostToDevice));
       
 //   printf("test1\n");

    // 矩阵计算
    cublasHandle_t handle;
    cublasStatus_t status;
    status = cublasCreate(&handle);

    cublasSgemmGroupedBatched(handle, transa_array, transb_array,//非转置
                     m, n, k,
                     alpha,
                     d_A_array, lda,
                     d_B_array, ldb,
                     beta,
                     d_C_array, ldc,
                     batchCount,
                     group_size);
 //   status =cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,//非转置
  //                          m, n, k, 
   //                         &alpha, 
    //                        d_A_array, lda,
     //                       d_B_array, ldb, 
      //                      &beta, 
       //                     d_C_array, ldc, 
        //                    batchCount);
  // cublasSgemmBatched();
   //cublasSgemmGroupedBatched();
  //  printf("test2\n");
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        exit(1);
    }
    cudaDeviceSynchronize();
  //  printf("test3\n");

   // printf("test4\n");
    // 释放空间
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));

  //  printf("Result: \n");
    //printMatrix(C);
return C;
}
