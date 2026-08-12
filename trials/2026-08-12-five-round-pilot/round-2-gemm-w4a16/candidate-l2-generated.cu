resource_usage {}
primary <bound method JITKernel._primary_resource_usage of <tilelang.jit.kernel.JITKernel object at 0x73689840cfe0>>
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ activation, half_t* __restrict__ output, const uchar* __restrict__ packed_weight, const float* __restrict__ weight_scale, const uchar* __restrict__ weight_zero);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ activation, half_t* __restrict__ output, const uchar* __restrict__ packed_weight, const float* __restrict__ weight_scale, const uchar* __restrict__ weight_zero) {
  float accum[1];
  extern __shared__ __align__(1024) half_t activation_shared[];
  float scales[1];
  float zeros[1];
  uchar packed_weight_local_cast[8];
  int packed_values[8];
  float products_low[8];
  float products_high[8];
  float partial_low[1];
  float partial_high[1];
  accum[0] = 0x0p+0f/*0.000000e+00*/;
  for (int group = 0; group < 64; ++group) {
    __syncthreads();
    activation_shared[((int)threadIdx.x)] = activation[((group * 128) + ((int)threadIdx.x))];
    scales[0] = weight_scale[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + group)];
    zeros[0] = ((float)weight_zero[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + group)]);
    *(uint2*)(packed_weight_local_cast + 0) = *(uint2*)(packed_weight + ((((((int)blockIdx.x) * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (group * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    for (int i = 0; i < 2; ++i) {
      int4 __1;
      uint v_ = *(uint*)(packed_weight_local_cast + (i * 4));
      __1.x = (int)(((unsigned char)(v_ >> 0)));
      __1.y = (int)(((unsigned char)(v_ >> 8)));
      __1.z = (int)(((unsigned char)(v_ >> 16)));
      __1.w = (int)(((unsigned char)(v_ >> 24)));
      *(int4*)(packed_values + (i * 4)) = __1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      int low = (packed_values[i_1] & 15);
      half_t dequant_low = ((half_t)((((float)low) - zeros[0]) * scales[0]));
      products_low[i_1] = (((float)activation_shared[(((((int)threadIdx.x) & 7) * 16) + (i_1 * 2))]) * ((float)dequant_low));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      int high = (packed_values[i_2] >> 4);
      half_t dequant_high = ((half_t)((((float)high) - zeros[0]) * scales[0]));
      products_high[i_2] = (((float)activation_shared[((((((int)threadIdx.x) & 7) * 16) + (i_2 * 2)) + 1)]) * ((float)dequant_high));
    }
    partial_low[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 8; ++rv) {
      partial_low[0] = (partial_low[0] + products_low[rv]);
    }
    partial_low[0] = tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<128>>::run(partial_low[0]);
    partial_high[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
      partial_high[0] = (partial_high[0] + products_high[rv_1]);
    }
    partial_high[0] = tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<128>>::run(partial_high[0]);
    accum[0] = ((accum[0] + partial_low[0]) + partial_high[0]);
  }
  if ((((int)threadIdx.x) % 8) == 0) {
    output[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))] = ((half_t)accum[0]);
  }
}
