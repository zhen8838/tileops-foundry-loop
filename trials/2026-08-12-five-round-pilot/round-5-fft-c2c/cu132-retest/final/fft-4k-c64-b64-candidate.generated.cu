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

extern "C" __global__ void _fft_lut_main_kernel(const float* __restrict__ lut_imag, const float* __restrict__ lut_real, const float* __restrict__ x_pair, float* __restrict__ y_imag, float* __restrict__ y_real);
extern "C" __global__ void _fft_lut_main_kernel_1(const float* __restrict__ lut_imag, const float* __restrict__ lut_real, const float* __restrict__ y_imag, float* __restrict__ y_pair, const float* __restrict__ y_real);
extern "C" __global__ void __launch_bounds__(256, 1) _fft_lut_main_kernel(const float* __restrict__ lut_imag, const float* __restrict__ lut_real, const float* __restrict__ x_pair, float* __restrict__ y_imag, float* __restrict__ y_real) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* smem_i = ((void*)((char*)buf_dyn_shmem + 0));
  void* smem_r = ((void*)((char*)buf_dyn_shmem + 2048));
  int rev_idx = 0;
  int temp = 0;
  int rev_idx_1 = 0;
  int temp_1 = 0;
  rev_idx = 0;
  temp = ((((int)blockIdx.x) * 512) + ((int)threadIdx.x));
  for (int _bit = 0; _bit < 12; ++_bit) {
    rev_idx = ((rev_idx << 1) | (temp & 1));
    temp = (temp >> 1);
  }
  float condval;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval = x_pair[((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx) * (int64_t)2))];
  } else {
    condval = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_r)[((int)threadIdx.x)] = condval;
  float condval_1;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval_1 = x_pair[(((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx) * (int64_t)2)) + (int64_t)1)];
  } else {
    condval_1 = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_i)[((int)threadIdx.x)] = condval_1;
  rev_idx_1 = 0;
  temp_1 = (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256);
  for (int _bit_1 = 0; _bit_1 < 12; ++_bit_1) {
    rev_idx_1 = ((rev_idx_1 << 1) | (temp_1 & 1));
    temp_1 = (temp_1 >> 1);
  }
  float condval_2;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_2 = x_pair[((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx_1) * (int64_t)2))];
  } else {
    condval_2 = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_r)[(((int)threadIdx.x) + 256)] = condval_2;
  float condval_3;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_3 = x_pair[(((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx_1) * (int64_t)2)) + (int64_t)1)];
  } else {
    condval_3 = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_i)[(((int)threadIdx.x) + 256)] = condval_3;
  __syncthreads();
  for (int s = 0; s < 9; ++s) {
    float tw_r = lut_real[(((1 << s) + (((int)threadIdx.x) % (1 << s))) - 1)];
    float tw_i = lut_imag[(((1 << s) + (((int)threadIdx.x) % (1 << s))) - 1)];
    float u_r = ((float*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    float u_i = ((float*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    float v_r = ((float*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    float v_i = ((float*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    float t_r = ((v_r * tw_r) - (v_i * tw_i));
    float t_i = ((v_r * tw_i) + (v_i * tw_r));
    __syncthreads();
    ((float*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_r + ((v_r * tw_r) - (v_i * tw_i)));
    ((float*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_i + ((v_r * tw_i) + (v_i * tw_r)));
    ((float*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_r - ((v_r * tw_r) - (v_i * tw_i)));
    ((float*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_i - ((v_r * tw_i) + (v_i * tw_r)));
    __syncthreads();
  }
  float value_real = ((float*)smem_r)[((int)threadIdx.x)];
  float value_imag = ((float*)smem_i)[((int)threadIdx.x)];
  y_real[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x))] = value_real;
  y_imag[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x))] = value_imag;
  float value_real_1 = ((float*)smem_r)[(((int)threadIdx.x) + 256)];
  float value_imag_1 = ((float*)smem_i)[(((int)threadIdx.x) + 256)];
  y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x)) + 256)] = value_real_1;
  y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x)) + 256)] = value_imag_1;
}

extern "C" __global__ void __launch_bounds__(256, 1) _fft_lut_main_kernel_1(const float* __restrict__ lut_imag, const float* __restrict__ lut_real, const float* __restrict__ y_imag, float* __restrict__ y_pair, const float* __restrict__ y_real) {
  float pair[2];
  float pair_1[2];
  float pair_2[2];
  float pair_3[2];
  float pair_4[2];
  float pair_5[2];
  float pair_6[2];
  float pair_7[2];
  float x0_r = y_real[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x))];
  float x0_i = y_imag[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x))];
  float x1_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 512)];
  float x1_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 512)];
  float x2_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1024)];
  float x2_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1024)];
  float x3_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1536)];
  float x3_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1536)];
  float x4_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2048)];
  float x4_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2048)];
  float x5_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2560)];
  float x5_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2560)];
  float x6_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3072)];
  float x6_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3072)];
  float x7_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3584)];
  float x7_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3584)];
  float w1_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 511)];
  float w1_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 511)];
  float a0_r = (x0_r + ((x1_r * w1_r) - (x1_i * w1_i)));
  float a0_i = (x0_i + ((x1_r * w1_i) + (x1_i * w1_r)));
  float a1_r = (x0_r - ((x1_r * w1_r) - (x1_i * w1_i)));
  float a1_i = (x0_i - ((x1_r * w1_i) + (x1_i * w1_r)));
  float a2_r = (x2_r + ((x3_r * w1_r) - (x3_i * w1_i)));
  float a2_i = (x2_i + ((x3_r * w1_i) + (x3_i * w1_r)));
  float a3_r = (x2_r - ((x3_r * w1_r) - (x3_i * w1_i)));
  float a3_i = (x2_i - ((x3_r * w1_i) + (x3_i * w1_r)));
  float a4_r = (x4_r + ((x5_r * w1_r) - (x5_i * w1_i)));
  float a4_i = (x4_i + ((x5_r * w1_i) + (x5_i * w1_r)));
  float a5_r = (x4_r - ((x5_r * w1_r) - (x5_i * w1_i)));
  float a5_i = (x4_i - ((x5_r * w1_i) + (x5_i * w1_r)));
  float a6_r = (x6_r + ((x7_r * w1_r) - (x7_i * w1_i)));
  float a6_i = (x6_i + ((x7_r * w1_i) + (x7_i * w1_r)));
  float a7_r = (x6_r - ((x7_r * w1_r) - (x7_i * w1_i)));
  float a7_i = (x6_i - ((x7_r * w1_i) + (x7_i * w1_r)));
  float w2a_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1023)];
  float w2a_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1023)];
  float w2b_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1535)];
  float w2b_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1535)];
  float b0_r = ((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i)));
  float b0_i = ((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r)));
  float b2_r = ((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i)));
  float b2_i = ((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r)));
  float b1_r = ((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i)));
  float b1_i = ((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r)));
  float b3_r = ((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i)));
  float b3_i = ((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r)));
  float b4_r = ((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i)));
  float b4_i = ((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r)));
  float b6_r = ((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i)));
  float b6_i = ((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r)));
  float b5_r = ((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i)));
  float b5_i = ((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r)));
  float b7_r = ((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i)));
  float b7_i = ((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r)));
  float w3a_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2047)];
  float w3a_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2047)];
  float w3b_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2559)];
  float w3b_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2559)];
  float w3c_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3071)];
  float w3c_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3071)];
  float w3d_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3583)];
  float w3d_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3583)];
  float value_real = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  float value_imag = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  float value_real_1 = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  float value_imag_1 = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  float value_real_2 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  float value_imag_2 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  float value_real_3 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  float value_imag_3 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  float value_real_4 = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  float value_imag_4 = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  float value_real_5 = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  float value_imag_5 = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  float value_real_6 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  float value_imag_6 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  float value_real_7 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  float value_imag_7 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  pair[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  pair[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  *(float2*)(y_pair + (((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2))) = *(float2*)(pair + 0);
  pair_1[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  pair_1[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 1024)) = *(float2*)(pair_1 + 0);
  pair_2[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  pair_2[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 2048)) = *(float2*)(pair_2 + 0);
  pair_3[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  pair_3[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 3072)) = *(float2*)(pair_3 + 0);
  pair_4[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  pair_4[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 4096)) = *(float2*)(pair_4 + 0);
  pair_5[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  pair_5[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 5120)) = *(float2*)(pair_5 + 0);
  pair_6[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  pair_6[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 6144)) = *(float2*)(pair_6 + 0);
  pair_7[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  pair_7[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  *(float2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 7168)) = *(float2*)(pair_7 + 0);
}
