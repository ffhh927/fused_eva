
// 定义EVA算法的核心计算过程，根据eva.py中的原理进行计算
// TensorListMetadata包含的参数：[ma(激活值),mg(pre-grad)]->v,grad（g), (vg_sum,v_sum,g_sum)
// 计算需要的  lr, damping, kl_clip,  
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_eva.cuh"

#define BLOCK_SIZE 512
#define ILP 4
using MATH_T = float;
using scalar_t_0 = at::BFloat16;
template<typename T, typename FULL_T, typename index_t>
struct EvaFunctor
{
   __device__ __forceinline__ void operator()(
    index_t chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const float lr,
    const float damping,
    const float imm)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    FULL_T* g = (FULL_T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[1][tensor_loc];
    v += chunk_idx*chunk_size;

    FULL_T* f = (FULL_T*)tl.addresses[2][tensor_loc];
    f += chunk_idx*chunk_size;


    n -= chunk_idx*chunk_size;
    
       // if (blockIdx.x == 0 && threadIdx.x == 0) {
       //     printf("尝试输出\n");
     //       printf("g = %f, v = %f, f = %f\n",g[0],v[0],f[0]);
   //     }
    // see note in multi_tensor_scale_kernel.cu
    for(index_t i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_f[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_f[ii] = f[i];
          r_v[ii] = v[i];
        //  printf("%f\n",g[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_f[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll

      for(int ii = 0; ii < ILP; ii++)
      {
        //r_v[ii] = r_ma[ii] * r_mg[ii] *(-1 * gtgrad * r_ma[ii] / (ata * gtg + damping));
        //r_v[ii] = r_ma[ii] * atg * -1 * gtgrad * / (ata * gtg + damping);
        r_v[ii] = r_g[ii] + r_f[ii] * imm;
        r_v[ii] = r_v[ii] / damping;
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
            v[i] = r_v[ii];
        }
      }
      
    }
    
       // if (blockIdx.x == 0 && threadIdx.x == 0) {
       //     printf("尝试输出\n");
     //       printf("g = %f, v = %f, f = %f\n",g[0],v[0],f[0]);
   //     }
  }
};



// 定义一个函数，用于启动EVA算法的核心计算过程

void multi_tensor_eva_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    const float lr,
    const float damping,
    const float imm)
{


    // 启动核函数
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        EvaFunctor<scalar_t_0, float, int32_t>(),
        lr,
        damping,
        imm);
        

}

