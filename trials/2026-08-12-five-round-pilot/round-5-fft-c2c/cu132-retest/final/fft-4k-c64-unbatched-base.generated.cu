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

extern "C" __global__ void _fft_lut_main_kernel(const float* __restrict__ x_imag, const float* __restrict__ x_real, float* __restrict__ y_imag, float* __restrict__ y_real);
extern "C" __global__ void _fft_lut_main_kernel_1(const float* __restrict__ lut_imag, const float* __restrict__ lut_real, const float* __restrict__ y_imag, float* __restrict__ y_pair, const float* __restrict__ y_real);
extern "C" __global__ void __launch_bounds__(512, 1) _fft_lut_main_kernel(const float* __restrict__ x_imag, const float* __restrict__ x_real, float* __restrict__ y_imag, float* __restrict__ y_real) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* smem_i = ((void*)((char*)buf_dyn_shmem + 0));
  void* smem_r = ((void*)((char*)buf_dyn_shmem + 4096));
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
  float condval;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval = x_real[rev_idx];
  } else {
    condval = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_r)[((int)threadIdx.x)] = condval;
  float condval_1;
  if (((0 <= rev_idx) && (rev_idx < 4096))) {
    condval_1 = x_imag[rev_idx];
  } else {
    condval_1 = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_i)[((int)threadIdx.x)] = condval_1;
  rev_idx_1 = 0;
  temp_1 = (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + 512);
  for (int _bit_1 = 0; _bit_1 < 12; ++_bit_1) {
    rev_idx_1 = ((rev_idx_1 << 1) | (temp_1 & 1));
    temp_1 = (temp_1 >> 1);
  }
  float condval_2;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_2 = x_real[rev_idx_1];
  } else {
    condval_2 = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_r)[(((int)threadIdx.x) + 512)] = condval_2;
  float condval_3;
  if (((0 <= rev_idx_1) && (rev_idx_1 < 4096))) {
    condval_3 = x_imag[rev_idx_1];
  } else {
    condval_3 = 0x0p+0f/*0.000000e+00*/;
  }
  ((float*)smem_i)[(((int)threadIdx.x) + 512)] = condval_3;
  __syncthreads();
  for (int s = 0; s < 10; ++s) {
    double angle = ((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))));
    float tw_r = cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))));
    float tw_i = sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))));
    float u_r = ((float*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    float u_i = ((float*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))];
    float v_r = ((float*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    float v_i = ((float*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))];
    float t_r = ((v_r * cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))) - (v_i * sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))));
    float t_i = ((v_r * sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))) + (v_i * cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))));
    __syncthreads();
    ((float*)smem_r)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_r + ((v_r * cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))) - (v_i * sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))))));
    ((float*)smem_i)[(((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s)))] = (u_i + ((v_r * sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))) + (v_i * cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))))));
    ((float*)smem_r)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_r - ((v_r * cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))) - (v_i * sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))))));
    ((float*)smem_i)[((((((int)threadIdx.x) / (1 << s)) * (1 << (s + 1))) + (((int)threadIdx.x) % (1 << s))) + (1 << s))] = (u_i - ((v_r * sinf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1))))))) + (v_i * cosf(((float)((-0x1.921fb54442d18p+2/*-6.283185e+00*/ * ((double)(((int)threadIdx.x) % (1 << s)))) / ((double)(1 << (s + 1)))))))));
    __syncthreads();
  }
  float value_real = ((float*)smem_r)[((int)threadIdx.x)];
  float value_imag = ((float*)smem_i)[((int)threadIdx.x)];
  y_real[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = value_real;
  y_imag[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = value_imag;
  float value_real_1 = ((float*)smem_r)[(((int)threadIdx.x) + 512)];
  float value_imag_1 = ((float*)smem_i)[(((int)threadIdx.x) + 512)];
  y_real[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + 512)] = value_real_1;
  y_imag[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + 512)] = value_imag_1;
}

extern "C" __global__ void __launch_bounds__(512, 1) _fft_lut_main_kernel_1(const float* __restrict__ lut_imag, const float* __restrict__ lut_real, const float* __restrict__ y_imag, float* __restrict__ y_pair, const float* __restrict__ y_real) {
  float pair[2];
  float pair_1[2];
  float pair_2[2];
  float pair_3[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float a_r = y_real[((i * 512) + ((int)threadIdx.x))];
    float a_i = y_imag[((i * 512) + ((int)threadIdx.x))];
    float b_r = y_real[(((i * 512) + ((int)threadIdx.x)) + 1024)];
    float b_i = y_imag[(((i * 512) + ((int)threadIdx.x)) + 1024)];
    float c_r = y_real[(((i * 512) + ((int)threadIdx.x)) + 2048)];
    float c_i = y_imag[(((i * 512) + ((int)threadIdx.x)) + 2048)];
    float d_r = y_real[(((i * 512) + ((int)threadIdx.x)) + 3072)];
    float d_i = y_imag[(((i * 512) + ((int)threadIdx.x)) + 3072)];
    float w1_r = lut_real[(((i * 512) + ((int)threadIdx.x)) + 1023)];
    float w1_i = lut_imag[(((i * 512) + ((int)threadIdx.x)) + 1023)];
    float w2_r = lut_real[(((i * 512) + ((int)threadIdx.x)) + 2047)];
    float w2_i = lut_imag[(((i * 512) + ((int)threadIdx.x)) + 2047)];
    float ap_r = (a_r + ((b_r * w1_r) - (b_i * w1_i)));
    float ap_i = (a_i + ((b_r * w1_i) + (b_i * w1_r)));
    float bp_r = (a_r - ((b_r * w1_r) - (b_i * w1_i)));
    float bp_i = (a_i - ((b_r * w1_i) + (b_i * w1_r)));
    float cp_r = (c_r + ((d_r * w1_r) - (d_i * w1_i)));
    float cp_i = (c_i + ((d_r * w1_i) + (d_i * w1_r)));
    float dp_r = (c_r - ((d_r * w1_r) - (d_i * w1_i)));
    float dp_i = (c_i - ((d_r * w1_i) + (d_i * w1_r)));
    float tc_r = (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i));
    float tc_i = (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r));
    float td2_r = (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0f/*-1.000000e+00*/)));
    float td2_i = (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0f/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i));
    float value_real = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    float value_imag = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    pair[0] = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair[1] = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) + (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    *(float2*)(y_pair + ((i * 1024) + (((int)threadIdx.x) * 2))) = *(float2*)(pair + 0);
    float value_real_1 = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    float value_imag_1 = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    pair_1[0] = ((a_r + ((b_r * w1_r) - (b_i * w1_i))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_r) - ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair_1[1] = ((a_i + ((b_r * w1_i) + (b_i * w1_r))) - (((c_r + ((d_r * w1_r) - (d_i * w1_i))) * w2_i) + ((c_i + ((d_r * w1_i) + (d_i * w1_r))) * w2_r)));
    *(float2*)(y_pair + (((i * 1024) + (((int)threadIdx.x) * 2)) + 4096)) = *(float2*)(pair_1 + 0);
    float value_real_2 = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0f/*-1.000000e+00*/))));
    float value_imag_2 = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0f/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair_2[0] = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0f/*-1.000000e+00*/))));
    pair_2[1] = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) + (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0f/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    *(float2*)(y_pair + (((i * 1024) + (((int)threadIdx.x) * 2)) + 2048)) = *(float2*)(pair_2 + 0);
    float value_real_3 = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0f/*-1.000000e+00*/))));
    float value_imag_3 = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0f/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    pair_3[0] = ((a_r - ((b_r * w1_r) - (b_i * w1_i))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * w2_i) - ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * (w2_r * -0x1p+0f/*-1.000000e+00*/))));
    pair_3[1] = ((a_i - ((b_r * w1_i) + (b_i * w1_r))) - (((c_r - ((d_r * w1_r) - (d_i * w1_i))) * (w2_r * -0x1p+0f/*-1.000000e+00*/)) + ((c_i - ((d_r * w1_i) + (d_i * w1_r))) * w2_i)));
    *(float2*)(y_pair + (((i * 1024) + (((int)threadIdx.x) * 2)) + 6144)) = *(float2*)(pair_3 + 0);
  }
}
