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

extern "C" __global__ void fft_radix2_kernel(const int* __restrict__ bit_reverse, float* __restrict__ output, const float* __restrict__ twiddle_imag, const float* __restrict__ twiddle_real, const float* __restrict__ x_imag, const float* __restrict__ x_real);
extern "C" __global__ void __launch_bounds__(256, 1) fft_radix2_kernel(const int* __restrict__ bit_reverse, float* __restrict__ output, const float* __restrict__ twiddle_imag, const float* __restrict__ twiddle_real, const float* __restrict__ x_imag, const float* __restrict__ x_real) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* shared_i = ((void*)((char*)buf_dyn_shmem + 0));
  void* shared_r = ((void*)((char*)buf_dyn_shmem + 16384));
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int vec_s = 0; vec_s < 4; ++vec_s) {
      int src = bit_reverse[(((i * 1024) + (((int)threadIdx.x) * 4)) + vec_s)];
      float condval;
      if (((0 <= src) && (src < 4096))) {
        condval = x_real[((int64_t)src)];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)shared_r)[(((i * 1024) + (((int)threadIdx.x) * 4)) + vec_s)] = condval;
      float condval_1;
      if (((0 <= src) && (src < 4096))) {
        condval_1 = x_imag[((int64_t)src)];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)shared_i)[(((i * 1024) + (((int)threadIdx.x) * 4)) + vec_s)] = condval_1;
    }
  }
  __syncthreads();
  for (int stage = 0; stage < 12; ++stage) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      __syncthreads();
      float even_r = ((float*)shared_r)[((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage)))];
      float even_i = ((float*)shared_i)[((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage)))];
      float odd_r = ((float*)shared_r)[(((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage))) + (1 << stage))];
      float odd_i = ((float*)shared_i)[(((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage))) + (1 << stage))];
      float tw_r = twiddle_real[(((1 << stage) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage))) - 1)];
      float tw_i = twiddle_imag[(((1 << stage) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage))) - 1)];
      float prod_r = ((odd_r * tw_r) - (odd_i * tw_i));
      float prod_i = ((odd_r * tw_i) + (odd_i * tw_r));
      __syncthreads();
      ((float*)shared_r)[((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage)))] = (even_r + ((odd_r * tw_r) - (odd_i * tw_i)));
      ((float*)shared_i)[((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage)))] = (even_i + ((odd_r * tw_i) + (odd_i * tw_r)));
      ((float*)shared_r)[(((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage))) + (1 << stage))] = (even_r - ((odd_r * tw_r) - (odd_i * tw_i)));
      ((float*)shared_i)[(((((((i_1 * 256) + ((int)threadIdx.x)) / (1 << stage)) * (1 << stage)) * 2) + (((i_1 * 256) + ((int)threadIdx.x)) % (1 << stage))) + (1 << stage))] = (even_i - ((odd_r * tw_i) + (odd_i * tw_r)));
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    output[((i_2 * 512) + (((int)threadIdx.x) * 2))] = ((float*)shared_r)[((i_2 * 256) + ((int)threadIdx.x))];
    output[(((i_2 * 512) + (((int)threadIdx.x) * 2)) + 1)] = ((float*)shared_i)[((i_2 * 256) + ((int)threadIdx.x))];
  }
}
