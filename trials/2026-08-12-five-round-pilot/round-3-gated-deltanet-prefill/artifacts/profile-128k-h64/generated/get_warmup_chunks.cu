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

extern "C" __global__ void tilelang_get_warmup_chunks_kernel_kernel(const int* __restrict__ cu_seqlens, signed char* __restrict__ fallback_mask, const bfloat16_t* __restrict__ g, const signed char* __restrict__ ht_mask, int* __restrict__ num_warmup_chunks, int batch_size, int num_tokens);
extern "C" __global__ void __launch_bounds__(64, 1) tilelang_get_warmup_chunks_kernel_kernel(const int* __restrict__ cu_seqlens, signed char* __restrict__ fallback_mask, const bfloat16_t* __restrict__ g, const signed char* __restrict__ ht_mask, int* __restrict__ num_warmup_chunks, int batch_size, int num_tokens) {
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int num_iters = 0;
  float g_cumsum[1];
  int n_fragment[1];
  signed char f_fragment[1];
  float g_fragment[1];
  if ((bool)ht_mask[((int64_t)((int)blockIdx.x))]) {
    num_warmup_chunks[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] = 0;
  } else {
    seq_start_idx = cu_seqlens[((int64_t)((int)blockIdx.x))];
    seq_end_idx = cu_seqlens[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];
    num_iters = ((seq_end_idx - seq_start_idx) >> 6);
    g_cumsum[0] = 0x0p+0f/*0.000000e+00*/;
    n_fragment[0] = num_iters;
    f_fragment[0] = (signed char)1;
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      bfloat16_t condval;
      if (((1 <= (seq_end_idx - (i_s * 64))) && ((seq_end_idx - (i_s * 64)) <= num_tokens))) {
        condval = g[((((((int64_t)seq_end_idx) * (int64_t)64) + ((int64_t)((int)threadIdx.x))) - (((int64_t)i_s) * (int64_t)4096)) - (int64_t)64)];
      } else {
        condval = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      g_fragment[0] = ((float)condval);
      g_cumsum[0] = (g_cumsum[0] + g_fragment[0]);
      if ((g_cumsum[0] < -0x1.4p+3f/*-1.000000e+01*/) && (n_fragment[0] == num_iters)) {
        n_fragment[0] = (i_s + 1);
        f_fragment[0] = (signed char)0;
      }
    }
    num_warmup_chunks[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] = n_fragment[0];
    fallback_mask[((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x)))] = ((signed char)((bool)f_fragment[0]));
  }
}
