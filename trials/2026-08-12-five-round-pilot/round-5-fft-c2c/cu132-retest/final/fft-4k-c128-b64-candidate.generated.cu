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

extern "C" __global__ void _fft_lut_main_kernel(const double* __restrict__ lut_imag, const double* __restrict__ lut_real, const double* __restrict__ x_pair, double* __restrict__ y_imag, double* __restrict__ y_real);
extern "C" __global__ void _fft_lut_main_kernel_1(const double* __restrict__ lut_imag, const double* __restrict__ lut_real, const double* __restrict__ y_imag, double* __restrict__ y_pair, const double* __restrict__ y_real);
extern "C" __global__ void __launch_bounds__(256, 1) _fft_lut_main_kernel(const double* __restrict__ lut_imag, const double* __restrict__ lut_real, const double* __restrict__ x_pair, double* __restrict__ y_imag, double* __restrict__ y_real) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* smem_i = ((void*)((char*)buf_dyn_shmem + 0));
  void* smem_r = ((void*)((char*)buf_dyn_shmem + 4096));
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
  double condval;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval = x_pair[((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx) * (int64_t)2))];
  } else {
    condval = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_r)[((int)threadIdx.x)] = condval;
  double condval_1;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval_1 = x_pair[(((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx) * (int64_t)2)) + (int64_t)1)];
  } else {
    condval_1 = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_i)[((int)threadIdx.x)] = condval_1;
  rev_idx_1 = 0;
  temp_1 = (((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256);
  for (int _bit_1 = 0; _bit_1 < 12; ++_bit_1) {
    rev_idx_1 = ((rev_idx_1 << 1) | (temp_1 & 1));
    temp_1 = (temp_1 >> 1);
  }
  double condval_2;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_2 = x_pair[((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx_1) * (int64_t)2))];
  } else {
    condval_2 = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_r)[(((int)threadIdx.x) + 256)] = condval_2;
  double condval_3;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_3 = x_pair[(((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)rev_idx_1) * (int64_t)2)) + (int64_t)1)];
  } else {
    condval_3 = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_i)[(((int)threadIdx.x) + 256)] = condval_3;
  __syncthreads();
  for (int s = 0; s < 9; ++s) {
    double tw_r = lut_real[(((1 << s) + (((int)threadIdx.x) % (1 << s))) - 1)];
    double tw_i = lut_imag[(((1 << s) + (((int)threadIdx.x) % (1 << s))) - 1)];
    double u_r = ((double*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    double u_i = ((double*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    double v_r = ((double*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    double v_i = ((double*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    double t_r = ((v_r * tw_r) - (v_i * tw_i));
    double t_i = ((v_r * tw_i) + (v_i * tw_r));
    __syncthreads();
    ((double*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_r + ((v_r * tw_r) - (v_i * tw_i)));
    ((double*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_i + ((v_r * tw_i) + (v_i * tw_r)));
    ((double*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_r - ((v_r * tw_r) - (v_i * tw_i)));
    ((double*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_i - ((v_r * tw_i) + (v_i * tw_r)));
    __syncthreads();
  }
  double value_real = ((double*)smem_r)[((int)threadIdx.x)];
  double value_imag = ((double*)smem_i)[((int)threadIdx.x)];
  y_real[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x))] = value_real;
  y_imag[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x))] = value_imag;
  double value_real_1 = ((double*)smem_r)[(((int)threadIdx.x) + 256)];
  double value_imag_1 = ((double*)smem_i)[(((int)threadIdx.x) + 256)];
  y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x)) + 256)] = value_real_1;
  y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 512)) + ((int)threadIdx.x)) + 256)] = value_imag_1;
}

extern "C" __global__ void __launch_bounds__(256, 1) _fft_lut_main_kernel_1(const double* __restrict__ lut_imag, const double* __restrict__ lut_real, const double* __restrict__ y_imag, double* __restrict__ y_pair, const double* __restrict__ y_real) {
  double pair[2];
  double pair_1[2];
  double pair_2[2];
  double pair_3[2];
  double pair_4[2];
  double pair_5[2];
  double pair_6[2];
  double pair_7[2];
  double x0_r = y_real[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x))];
  double x0_i = y_imag[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x))];
  double x1_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 512)];
  double x1_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 512)];
  double x2_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1024)];
  double x2_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1024)];
  double x3_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1536)];
  double x3_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 1536)];
  double x4_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2048)];
  double x4_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2048)];
  double x5_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2560)];
  double x5_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 2560)];
  double x6_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3072)];
  double x6_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3072)];
  double x7_r = y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3584)];
  double x7_i = y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 256)) + ((int)threadIdx.x)) + 3584)];
  double w1_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 511)];
  double w1_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 511)];
  double a0_r = (x0_r + ((x1_r * w1_r) - (x1_i * w1_i)));
  double a0_i = (x0_i + ((x1_r * w1_i) + (x1_i * w1_r)));
  double a1_r = (x0_r - ((x1_r * w1_r) - (x1_i * w1_i)));
  double a1_i = (x0_i - ((x1_r * w1_i) + (x1_i * w1_r)));
  double a2_r = (x2_r + ((x3_r * w1_r) - (x3_i * w1_i)));
  double a2_i = (x2_i + ((x3_r * w1_i) + (x3_i * w1_r)));
  double a3_r = (x2_r - ((x3_r * w1_r) - (x3_i * w1_i)));
  double a3_i = (x2_i - ((x3_r * w1_i) + (x3_i * w1_r)));
  double a4_r = (x4_r + ((x5_r * w1_r) - (x5_i * w1_i)));
  double a4_i = (x4_i + ((x5_r * w1_i) + (x5_i * w1_r)));
  double a5_r = (x4_r - ((x5_r * w1_r) - (x5_i * w1_i)));
  double a5_i = (x4_i - ((x5_r * w1_i) + (x5_i * w1_r)));
  double a6_r = (x6_r + ((x7_r * w1_r) - (x7_i * w1_i)));
  double a6_i = (x6_i + ((x7_r * w1_i) + (x7_i * w1_r)));
  double a7_r = (x6_r - ((x7_r * w1_r) - (x7_i * w1_i)));
  double a7_i = (x6_i - ((x7_r * w1_i) + (x7_i * w1_r)));
  double w2a_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1023)];
  double w2a_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1023)];
  double w2b_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1535)];
  double w2b_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 1535)];
  double b0_r = ((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i)));
  double b0_i = ((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r)));
  double b2_r = ((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i)));
  double b2_i = ((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r)));
  double b1_r = ((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i)));
  double b1_i = ((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r)));
  double b3_r = ((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i)));
  double b3_i = ((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r)));
  double b4_r = ((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i)));
  double b4_i = ((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r)));
  double b6_r = ((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i)));
  double b6_i = ((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r)));
  double b5_r = ((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i)));
  double b5_i = ((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r)));
  double b7_r = ((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i)));
  double b7_i = ((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r)));
  double w3a_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2047)];
  double w3a_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2047)];
  double w3b_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2559)];
  double w3b_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 2559)];
  double w3c_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3071)];
  double w3c_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3071)];
  double w3d_r = lut_real[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3583)];
  double w3d_i = lut_imag[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) + 3583)];
  double value_real = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  double value_imag = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  double value_real_1 = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  double value_imag_1 = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  double value_real_2 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  double value_imag_2 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  double value_real_3 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  double value_imag_3 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  double value_real_4 = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  double value_imag_4 = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  double value_real_5 = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  double value_imag_5 = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  double value_real_6 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  double value_imag_6 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  double value_real_7 = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  double value_imag_7 = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  pair[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  pair[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  *(double2*)(y_pair + (((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2))) = *(double2*)(pair + 0);
  pair_1[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  pair_1[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 1024)) = *(double2*)(pair_1 + 0);
  pair_2[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  pair_2[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) + ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 2048)) = *(double2*)(pair_2 + 0);
  pair_3[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  pair_3[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) + ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 3072)) = *(double2*)(pair_3 + 0);
  pair_4[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_i)));
  pair_4[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3a_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3a_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 4096)) = *(double2*)(pair_4 + 0);
  pair_5[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_i)));
  pair_5[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) + (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3b_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) + (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3b_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 5120)) = *(double2*)(pair_5 + 0);
  pair_6[0] = (((x0_r + ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_r) - ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_i))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_r) - (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_i)));
  pair_6[1] = (((x0_i + ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r + ((x3_r * w1_r) - (x3_i * w1_i))) * w2a_i) + ((x2_i + ((x3_r * w1_i) + (x3_i * w1_r))) * w2a_r))) - ((((x4_r + ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_r) - ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_i))) * w3c_i) + (((x4_i + ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r + ((x7_r * w1_r) - (x7_i * w1_i))) * w2a_i) + ((x6_i + ((x7_r * w1_i) + (x7_i * w1_r))) * w2a_r))) * w3c_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 6144)) = *(double2*)(pair_6 + 0);
  pair_7[0] = (((x0_r - ((x1_r * w1_r) - (x1_i * w1_i))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_r) - ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_i))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_r) - (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_i)));
  pair_7[1] = (((x0_i - ((x1_r * w1_i) + (x1_i * w1_r))) - (((x2_r - ((x3_r * w1_r) - (x3_i * w1_i))) * w2b_i) + ((x2_i - ((x3_r * w1_i) + (x3_i * w1_r))) * w2b_r))) - ((((x4_r - ((x5_r * w1_r) - (x5_i * w1_i))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_r) - ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_i))) * w3d_i) + (((x4_i - ((x5_r * w1_i) + (x5_i * w1_r))) - (((x6_r - ((x7_r * w1_r) - (x7_i * w1_i))) * w2b_i) + ((x6_i - ((x7_r * w1_i) + (x7_i * w1_r))) * w2b_r))) * w3d_r)));
  *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 2)) + 7168)) = *(double2*)(pair_7 + 0);
}
