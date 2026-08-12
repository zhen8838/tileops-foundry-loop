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

extern "C" __global__ void _fft_lut_main_kernel(const double* __restrict__ x_imag, const double* __restrict__ x_real, double* __restrict__ y_imag, double* __restrict__ y_real);
extern "C" __global__ void _fft_lut_main_kernel_1(const double* __restrict__ lut_imag, const double* __restrict__ lut_real, const double* __restrict__ y_imag, double* __restrict__ y_pair, const double* __restrict__ y_real);
extern "C" __global__ void __launch_bounds__(512, 1) _fft_lut_main_kernel(const double* __restrict__ x_imag, const double* __restrict__ x_real, double* __restrict__ y_imag, double* __restrict__ y_real) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* smem_i = ((void*)((char*)buf_dyn_shmem + 0));
  void* smem_r = ((void*)((char*)buf_dyn_shmem + 8192));
  int rev_idx = 0;
  int temp = 0;
  int rev_idx_1 = 0;
  int temp_1 = 0;
  rev_idx = 0;
  temp = ((((int)blockIdx.x) * 1024) + ((int)threadIdx.x));
  for (int _bit = 0; _bit < 12; ++_bit) {
    rev_idx = ((rev_idx << 1) | (temp & 1));
    temp = (temp >> 1);
  }
  double condval;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval = x_real[((((int64_t)((int)blockIdx.y)) * (int64_t)4096) + ((int64_t)rev_idx))];
  } else {
    condval = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_r)[((int)threadIdx.x)] = condval;
  double condval_1;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval_1 = x_imag[((((int64_t)((int)blockIdx.y)) * (int64_t)4096) + ((int64_t)rev_idx))];
  } else {
    condval_1 = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_i)[((int)threadIdx.x)] = condval_1;
  rev_idx_1 = 0;
  temp_1 = (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + 512);
  for (int _bit_1 = 0; _bit_1 < 12; ++_bit_1) {
    rev_idx_1 = ((rev_idx_1 << 1) | (temp_1 & 1));
    temp_1 = (temp_1 >> 1);
  }
  double condval_2;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_2 = x_real[((((int64_t)((int)blockIdx.y)) * (int64_t)4096) + ((int64_t)rev_idx_1))];
  } else {
    condval_2 = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_r)[(((int)threadIdx.x) + 512)] = condval_2;
  double condval_3;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_3 = x_imag[((((int64_t)((int)blockIdx.y)) * (int64_t)4096) + ((int64_t)rev_idx_1))];
  } else {
    condval_3 = 0x0p+0/*0.000000e+00*/;
  }
  ((double*)smem_i)[(((int)threadIdx.x) + 512)] = condval_3;
  __syncthreads();
  for (int s = 0; s < 10; ++s) {
    double angle = ((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))));
    double tw_r = cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))));
    double tw_i = sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))));
    double u_r = ((double*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    double u_i = ((double*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    double v_r = ((double*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    double v_i = ((double*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    double t_r = ((v_r * cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))) - (v_i * sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))));
    double t_i = ((v_r * sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))) + (v_i * cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))));
    __syncthreads();
    ((double*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_r + ((v_r * cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))) - (v_i * sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))));
    ((double*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_i + ((v_r * sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))) + (v_i * cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))));
    ((double*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_r - ((v_r * cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))) - (v_i * sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))));
    ((double*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_i - ((v_r * sin(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))) + (v_i * cos(((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))));
    __syncthreads();
  }
  double value_real = ((double*)smem_r)[((int)threadIdx.x)];
  double value_imag = ((double*)smem_i)[((int)threadIdx.x)];
  y_real[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = value_real;
  y_imag[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = value_imag;
  double value_real_1 = ((double*)smem_r)[(((int)threadIdx.x) + 512)];
  double value_imag_1 = ((double*)smem_i)[(((int)threadIdx.x) + 512)];
  y_real[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) + 512)] = value_real_1;
  y_imag[((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) + 512)] = value_imag_1;
}

extern "C" __global__ void __launch_bounds__(512, 1) _fft_lut_main_kernel_1(const double* __restrict__ lut_imag, const double* __restrict__ lut_real, const double* __restrict__ y_imag, double* __restrict__ y_pair, const double* __restrict__ y_real) {
  double pair[2];
  double pair_1[2];
  double pair_2[2];
  double pair_3[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    double a_r = y_real[(((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x))];
    double a_i = y_imag[(((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x))];
    double b_r = y_real[((((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x)) + 1024)];
    double b_i = y_imag[((((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x)) + 1024)];
    double c_r = y_real[((((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x)) + 2048)];
    double c_i = y_imag[((((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x)) + 2048)];
    double d_r = y_real[((((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x)) + 3072)];
    double d_i = y_imag[((((((int)blockIdx.y) * 4096) + (i * 512)) + ((int)threadIdx.x)) + 3072)];
    double w1_r = lut_real[(((i * 512) + ((int)threadIdx.x)) + 1023)];
    double w1_i = lut_imag[(((i * 512) + ((int)threadIdx.x)) + 1023)];
    double w2_r = lut_real[(((i * 512) + ((int)threadIdx.x)) + 2047)];
    double w2_i = lut_imag[(((i * 512) + ((int)threadIdx.x)) + 2047)];
    double ap_r = (a_r + ((b_r * w1_r) - (b_i * w1_i)));
    double ap_i = (a_i + ((b_r * w1_i) + (b_i * w1_r)));
    double bp_r = (a_r - ((b_r * w1_r) - (b_i * w1_i)));
    double bp_i = (a_i - ((b_r * w1_i) + (b_i * w1_r)));
    double cp_r = (c_r + ((d_r * w1_r) - (d_i * w1_i)));
    double cp_i = (c_i + ((d_r * w1_i) + (d_i * w1_r)));
    double dp_r = (c_r - ((d_r * w1_r) - (d_i * w1_i)));
    double dp_i = (c_i - ((d_r * w1_i) + (d_i * w1_r)));
    double tc_r = (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i));
    double tc_i = (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r));
    double td2_r = (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0/*-1.000000e+00*/)));
    double td2_i = (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i));
    double value_real = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    double value_imag = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    pair[0] = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair[1] = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    *(double2*)(y_pair + (((((int)blockIdx.y) * 8192) + (i * 1024)) + (((int)threadIdx.x) * 2))) = *(double2*)(pair + 0);
    double value_real_1 = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    double value_imag_1 = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    pair_1[0] = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair_1[1] = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (i * 1024)) + (((int)threadIdx.x) * 2)) + 4096)) = *(double2*)(pair_1 + 0);
    double value_real_2 = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0/*-1.000000e+00*/))));
    double value_imag_2 = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair_2[0] = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0/*-1.000000e+00*/))));
    pair_2[1] = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (i * 1024)) + (((int)threadIdx.x) * 2)) + 2048)) = *(double2*)(pair_2 + 0);
    double value_real_3 = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0/*-1.000000e+00*/))));
    double value_imag_3 = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair_3[0] = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0/*-1.000000e+00*/))));
    pair_3[1] = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    *(double2*)(y_pair + ((((((int)blockIdx.y) * 8192) + (i * 1024)) + (((int)threadIdx.x) * 2)) + 6144)) = *(double2*)(pair_3 + 0);
  }
}
