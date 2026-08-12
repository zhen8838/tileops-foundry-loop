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

extern "C" __global__ void tilelang_zero_initial_cp_h0_kernel_kernel(float* __restrict__ cp_h0, const int* __restrict__ seq_map_r2c, int cp_batch_size, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(128, 1) tilelang_zero_initial_cp_h0_kernel_kernel(float* __restrict__ cp_h0, const int* __restrict__ seq_map_r2c, int cp_batch_size, int raw_batch_size) {
  int condval;
  if (((((int)blockIdx.x) >> 8) <= raw_batch_size)) {
    condval = seq_map_r2c[(((int64_t)((int)blockIdx.x)) >> (int64_t)8)];
  } else {
    condval = 0;
  }
  int seq_start_idx = condval;
  if (0 <= seq_start_idx) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (seq_start_idx < cp_batch_size) {
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(cp_h0 + ((((((((int64_t)seq_start_idx) * (int64_t)1048576) + (((((int64_t)((int)blockIdx.x)) & (int64_t)255) >> (int64_t)2) * (int64_t)16384)) + (((int64_t)i) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      }
    }
  }
}
