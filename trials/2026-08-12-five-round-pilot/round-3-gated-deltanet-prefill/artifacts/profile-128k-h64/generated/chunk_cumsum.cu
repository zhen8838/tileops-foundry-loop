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

extern "C" __global__ void chunk_cumsum_bthd_kernel_kernel(const bfloat16_t* __restrict__ g, bfloat16_t* __restrict__ out);
extern "C" __global__ void __launch_bounds__(128, 1) chunk_cumsum_bthd_kernel_kernel(const bfloat16_t* __restrict__ g, bfloat16_t* __restrict__ out) {
  extern __shared__ __align__(1024) float acc_s[];
  if (((int)threadIdx.x) < 64) {
    acc_s[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    for (int i = 0; i < 64; ++i) {
      acc_s[((int)threadIdx.x)] = (acc_s[((int)threadIdx.x)] + ((float)g[(((((int)blockIdx.y) * 4096) + (i * 64)) + ((int)threadIdx.x))]));
      out[(((((int)blockIdx.y) * 4096) + (i * 64)) + ((int)threadIdx.x))] = ((bfloat16_t)acc_s[((int)threadIdx.x)]);
    }
  }
}
