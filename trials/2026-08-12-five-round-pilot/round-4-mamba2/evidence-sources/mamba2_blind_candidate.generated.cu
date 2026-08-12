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

extern "C" __global__ void mamba2_serial_kernel(const float* __restrict__ A, const bfloat16_t* __restrict__ B, const bfloat16_t* __restrict__ C, const float* __restrict__ dt, float* __restrict__ final_state, const bfloat16_t* __restrict__ x, float* __restrict__ y);
extern "C" __global__ void __launch_bounds__(128, 1) mamba2_serial_kernel(const float* __restrict__ A, const bfloat16_t* __restrict__ B, const bfloat16_t* __restrict__ C, const float* __restrict__ dt, float* __restrict__ final_state, const bfloat16_t* __restrict__ x, float* __restrict__ y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* state = ((void*)((char*)buf_dyn_shmem + 0));
  void* decay_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* delta_shared = ((void*)((char*)buf_dyn_shmem + 32784));
  bfloat16_t x_local_cast_1[4];
  bfloat16_t B_local_cast_2[4];
  float state_local_cast[4];
  float y_fragment[1];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(((float*)state) + ((i * 512) + (((int)threadIdx.x) * 4))) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  __syncthreads();
  for (int token = 0; token < 2048; ++token) {
    if (((int)threadIdx.x) == 0) {
      float condval;
      if ((0x1.4p+4f/*2.000000e+01*/ < dt[((token * 80) + ((int)blockIdx.y))])) {
        condval = dt[((token * 80) + ((int)blockIdx.y))];
      } else {
        condval = logf((0x1p+0f/*1.000000e+00*/ + expf(dt[((token * 80) + ((int)blockIdx.y))])));
      }
      ((float*)delta_shared)[0] = max(condval, 0x0p+0f/*0.000000e+00*/);
      ((float*)decay_shared)[0] = exp2f(((A[((int)blockIdx.y)] * ((float*)delta_shared)[0]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    }
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(uint2*)(x_local_cast_1 + 0) = make_uint2(__pack_nv_bfloat162(x[((((token * 5120) + (((int)blockIdx.y) * 64)) + (i_1 * 4)) + (((int)threadIdx.x) >> 5))], x[((((token * 5120) + (((int)blockIdx.y) * 64)) + (i_1 * 4)) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(x[((((token * 5120) + (((int)blockIdx.y) * 64)) + (i_1 * 4)) + (((int)threadIdx.x) >> 5))], x[((((token * 5120) + (((int)blockIdx.y) * 64)) + (i_1 * 4)) + (((int)threadIdx.x) >> 5))]));
      *(uint2*)(B_local_cast_2 + 0) = *(uint2*)(B + ((token * 128) + ((((int)threadIdx.x) & 31) * 4)));
      *(float4*)(state_local_cast + 0) = *(float4*)(((float*)state) + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
      float4 __1;
        float4 __2;
          float4 v_ = make_float4(((float*)decay_shared)[0], ((float*)decay_shared)[0], ((float*)decay_shared)[0], ((float*)decay_shared)[0]);
          float4 v__1 = *(float4*)(state_local_cast + 0);
          __2.x = (v_.x*v__1.x);
          __2.y = (v_.y*v__1.y);
          __2.z = (v_.z*v__1.z);
          __2.w = (v_.w*v__1.w);
        float4 __3;
          float4 __4;
            float4 v__2 = make_float4(((float*)delta_shared)[0], ((float*)delta_shared)[0], ((float*)delta_shared)[0], ((float*)delta_shared)[0]);
            float4 __5;
            uint2 v__3 = *(uint2*)(x_local_cast_1 + 0);
            ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            ((float2*)(&__5))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
            __4.x = (v__2.x*__5.x);
            __4.y = (v__2.y*__5.y);
            __4.z = (v__2.z*__5.z);
            __4.w = (v__2.w*__5.w);
          float4 __6;
          uint2 v__4 = *(uint2*)(B_local_cast_2 + 0);
          ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
          ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
          __3.x = (__4.x*__6.x);
          __3.y = (__4.y*__6.y);
          __3.z = (__4.z*__6.z);
          __3.w = (__4.w*__6.w);
        __1.x = (__2.x+__3.x);
        __1.y = (__2.y+__3.y);
        __1.z = (__2.z+__3.z);
        __1.w = (__2.w+__3.w);
      *(float4*)(state_local_cast + 0) = __1;
      *(float4*)(((float*)state) + ((i_1 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(state_local_cast + 0);
    }
    __syncthreads();
    y_fragment[0] = 0x0p+0f/*0.000000e+00*/;
    for (int n = 0; n < 128; ++n) {
      y_fragment[0] = (y_fragment[0] + (((float*)state)[(((((int)threadIdx.x) & 63) * 128) + n)] * ((float)C[((token * 128) + n)])));
    }
    if ((((int)threadIdx.x) >> 6) == 0) {
      y[(((token * 5120) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.x) & 63))] = y_fragment[0];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    *(float4*)(final_state + (((((int)blockIdx.y) * 8192) + (i_2 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)state) + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
  }
}
