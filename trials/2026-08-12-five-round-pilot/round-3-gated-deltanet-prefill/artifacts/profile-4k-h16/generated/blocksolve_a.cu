#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void prefill_blocksolve_A_bthd_tl_kernel(bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ beta, const bfloat16_t* __restrict__ g, const bfloat16_t* __restrict__ k);
extern "C" __global__ void __launch_bounds__(32, 1) prefill_blocksolve_A_bthd_tl_kernel(bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ beta, const bfloat16_t* __restrict__ g, const bfloat16_t* __restrict__ k) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* g_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* beta_s = ((void*)((char*)buf_dyn_shmem + 128));
  void* gate0_s = ((void*)((char*)buf_dyn_shmem + 256));
  void* k0 = ((void*)((char*)buf_dyn_shmem + 256));
  void* gate1_s = ((void*)((char*)buf_dyn_shmem + 384));
  void* gate2_s = ((void*)((char*)buf_dyn_shmem + 512));
  void* gate3_s = ((void*)((char*)buf_dyn_shmem + 640));
  void* beta_f_s = ((void*)((char*)buf_dyn_shmem + 768));
  void* a_s = ((void*)((char*)buf_dyn_shmem + 896));
  void* k1 = ((void*)((char*)buf_dyn_shmem + 2304));
  void* k2 = ((void*)((char*)buf_dyn_shmem + 4352));
  void* i_s = ((void*)((char*)buf_dyn_shmem + 6016));
  void* k3 = ((void*)((char*)buf_dyn_shmem + 6400));
  void* work_s = ((void*)((char*)buf_dyn_shmem + 8064));
  float G00[8];
  float G10[8];
  float G11[8];
  float G20[8];
  float G21[8];
  float G22[8];
  float G30[8];
  float G31[8];
  float G32[8];
  float G33[8];
  float tmp[8];
  bfloat16_t i_s_local_cast[2];
  bfloat16_t a_s_local_cast_1[2];
  bfloat16_t i_s_local_cast_2[2];
  bfloat16_t a_s_local_cast_3[2];
  bfloat16_t i_s_local_cast_4[2];
  bfloat16_t a_s_local_cast_5[2];
  bfloat16_t i_s_local_cast_6[2];
  bfloat16_t a_s_local_cast_7[2];
  bfloat16_t work_s_local_cast_8[2];
  bfloat16_t a_s_local_cast_9[2];
  bfloat16_t work_s_local_cast_10[2];
  bfloat16_t a_s_local_cast_11[2];
  bfloat16_t work_s_local_cast_12[2];
  bfloat16_t work_s_local_cast_13[2];
  bfloat16_t a_s_local_cast_14[2];
  bfloat16_t work_s_local_cast_15[2];
  bfloat16_t work_s_local_cast_16[2];
  bfloat16_t work_s_local_cast_17[2];
  bfloat16_t a_s_local_cast_18[2];
  bfloat16_t work_s_local_cast_19[2];
  bfloat16_t work_s_local_cast_20[2];
  bfloat16_t a_s_local_cast_21[2];
  bfloat16_t work_s_local_cast_22[2];
  bfloat16_t a_s_local_cast_23[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    ((bfloat16_t*)g_s)[((i * 32) + ((int)threadIdx.x))] = g[((((((int)blockIdx.z) * 1024) + (i * 512)) + (((int)threadIdx.x) * 16)) + ((int)blockIdx.y))];
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    ((bfloat16_t*)beta_s)[((i_1 * 32) + ((int)threadIdx.x))] = beta[((((((int)blockIdx.z) * 1024) + (i_1 * 512)) + (((int)threadIdx.x) * 16)) + ((int)blockIdx.y))];
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G00 + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G10 + (i_3 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G11 + (i_4 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G20 + (i_5 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G21 + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G22 + (i_7 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 2; ++i_8) {
    float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G30 + (i_8 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 2; ++i_9) {
    float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G31 + (i_9 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 2; ++i_10) {
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G32 + (i_10 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(G33 + (i_11 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  for (int kt = 0; kt < 2; ++kt) {
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)k0)[(((((i_12 * 256) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[((((((((int)blockIdx.z) * 131072) + (i_12 * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (kt * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)k1)[(((((i_13 * 256) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[(((((((((int)blockIdx.z) * 131072) + (i_13 * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (kt * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 32768)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)k2)[(((((i_14 * 256) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[(((((((((int)blockIdx.z) * 131072) + (i_14 * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (kt * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)k3)[(((((i_15 * 256) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_15 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[(((((((((int)blockIdx.z) * 131072) + (i_15 * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (kt * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 98304)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[8];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k0)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k0)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G00 + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G00 + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k1)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k0)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G10 + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G10 + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
    {
      bfloat16_t A_local_2[8];
      bfloat16_t B_local_2[8];
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k1)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k1)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G11 + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G11 + 4), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 4));
      }
    }
    {
      bfloat16_t A_local_3[8];
      bfloat16_t B_local_3[8];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k2)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k0)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G20 + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G20 + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
      }
    }
    {
      bfloat16_t A_local_4[8];
      bfloat16_t B_local_4[8];
      for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k2)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_4[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k1)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G21 + 0), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G21 + 4), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + 4));
      }
    }
    {
      bfloat16_t A_local_5[8];
      bfloat16_t B_local_5[8];
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k2)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_5[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k2)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_5[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G22 + 0), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G22 + 4), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + 4));
      }
    }
    {
      bfloat16_t A_local_6[8];
      bfloat16_t B_local_6[8];
      for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k3)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k0)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_6[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G30 + 0), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G30 + 4), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 4));
      }
    }
    {
      bfloat16_t A_local_7[8];
      bfloat16_t B_local_7[8];
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k3)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_7[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k1)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_7 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_7[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G31 + 0), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G31 + 4), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_7 + 4));
      }
    }
    {
      bfloat16_t A_local_8[8];
      bfloat16_t B_local_8[8];
      for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k3)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_8[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k2)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_8[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G32 + 0), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G32 + 4), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_8 + 4));
      }
    }
    {
      bfloat16_t A_local_9[8];
      bfloat16_t B_local_9[8];
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k3)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_9[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k3)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_9[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G33 + 0), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(G33 + 4), reinterpret_cast<const unsigned*>(A_local_9 + 0), reinterpret_cast<const unsigned*>(B_local_9 + 4));
      }
    }
  }
  float2 __1;
  uint1 v_ = *(uint1*)(((bfloat16_t*)g_s) + (((int)threadIdx.x) * 2));
  ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
  float2 g_val = __1;
  __syncthreads();
  float broadcast_var_10 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
  uint1 __2;
  float2 __3;
  float2 __4;
    float2 __5;
      float2 v__1 = make_float2(((float)((bfloat16_t*)g_s)[0]), ((float)((bfloat16_t*)g_s)[0]));
      __5.x = (g_val.x-v__1.x);
      __5.y = (g_val.y-v__1.y);
    float2 v__2 = make_float2(broadcast_var_10, broadcast_var_10);
    __4.x = (__5.x*v__2.x);
    __4.y = (__5.y*v__2.y);
  __3.x = exp2f(__4.x);
  __3.y = exp2f(__4.y);
  (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&__3))[0]);
  *(uint1*)(((bfloat16_t*)gate0_s) + (((int)threadIdx.x) * 2)) = __2;
  float broadcast_var_11 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
  uint1 __6;
  float2 __7;
  float2 __8;
    float2 __9;
      float2 v__3 = make_float2(((float)((bfloat16_t*)g_s)[16]), ((float)((bfloat16_t*)g_s)[16]));
      __9.x = (g_val.x-v__3.x);
      __9.y = (g_val.y-v__3.y);
    float2 v__4 = make_float2(broadcast_var_11, broadcast_var_11);
    __8.x = (__9.x*v__4.x);
    __8.y = (__9.y*v__4.y);
  __7.x = exp2f(__8.x);
  __7.y = exp2f(__8.y);
  (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&__7))[0]);
  *(uint1*)(((bfloat16_t*)gate1_s) + (((int)threadIdx.x) * 2)) = __6;
  float broadcast_var_12 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
  uint1 __10;
  float2 __11;
  float2 __12;
    float2 __13;
      float2 v__5 = make_float2(((float)((bfloat16_t*)g_s)[32]), ((float)((bfloat16_t*)g_s)[32]));
      __13.x = (g_val.x-v__5.x);
      __13.y = (g_val.y-v__5.y);
    float2 v__6 = make_float2(broadcast_var_12, broadcast_var_12);
    __12.x = (__13.x*v__6.x);
    __12.y = (__13.y*v__6.y);
  __11.x = exp2f(__12.x);
  __11.y = exp2f(__12.y);
  (reinterpret_cast<__nv_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&__11))[0]);
  *(uint1*)(((bfloat16_t*)gate2_s) + (((int)threadIdx.x) * 2)) = __10;
  float broadcast_var_13 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
  uint1 __14;
  float2 __15;
  float2 __16;
    float2 __17;
      float2 v__7 = make_float2(((float)((bfloat16_t*)g_s)[48]), ((float)((bfloat16_t*)g_s)[48]));
      __17.x = (g_val.x-v__7.x);
      __17.y = (g_val.y-v__7.y);
    float2 v__8 = make_float2(broadcast_var_13, broadcast_var_13);
    __16.x = (__17.x*v__8.x);
    __16.y = (__17.y*v__8.y);
  __15.x = exp2f(__16.x);
  __15.y = exp2f(__16.y);
  (reinterpret_cast<__nv_bfloat162*>(&__14))[0] = __float22bfloat162_rn(((float2*)(&__15))[0]);
  *(uint1*)(((bfloat16_t*)gate3_s) + (((int)threadIdx.x) * 2)) = __14;
  *(uint1*)(((bfloat16_t*)beta_f_s) + (((int)threadIdx.x) * 2)) = *(uint1*)(((bfloat16_t*)beta_s) + (((int)threadIdx.x) * 2));
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    float s00 = ((((float)((bfloat16_t*)beta_f_s)[((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2))]) * ((float)((bfloat16_t*)gate0_s)[((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2))])) / ((float)((bfloat16_t*)gate0_s)[((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1))]));
    float s11 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 16)]) * ((float)((bfloat16_t*)gate1_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 16)])) / ((float)((bfloat16_t*)gate1_s)[(((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 16)]));
    float s22 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 32)]) * ((float)((bfloat16_t*)gate2_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 32)])) / ((float)((bfloat16_t*)gate2_s)[(((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 32)]));
    float s33 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)]) * ((float)((bfloat16_t*)gate3_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)])) / ((float)((bfloat16_t*)gate3_s)[(((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 48)]));
    float s10 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 16)]) * ((float)((bfloat16_t*)gate0_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 16)])) / ((float)((bfloat16_t*)gate0_s)[((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1))]));
    float s20 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 32)]) * ((float)((bfloat16_t*)gate0_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 32)])) / ((float)((bfloat16_t*)gate0_s)[((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1))]));
    float s21 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 32)]) * ((float)((bfloat16_t*)gate1_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 32)])) / ((float)((bfloat16_t*)gate1_s)[(((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 16)]));
    float s30 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)]) * ((float)((bfloat16_t*)gate0_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)])) / ((float)((bfloat16_t*)gate0_s)[((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1))]));
    float s31 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)]) * ((float)((bfloat16_t*)gate1_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)])) / ((float)((bfloat16_t*)gate1_s)[(((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 16)]));
    float s32 = ((((float)((bfloat16_t*)beta_f_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)]) * ((float)((bfloat16_t*)gate2_s)[(((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) + 48)])) / ((float)((bfloat16_t*)gate2_s)[(((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 32)]));
    float condval;
    if ((((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) < ((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)))) {
      condval = ((G00[i_16] * -0x1p+0f/*-1.000000e+00*/) * s00);
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)a_s)[(((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1))] = ((bfloat16_t)condval);
    float condval_1;
    if ((((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) < ((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)))) {
      condval_1 = ((G11[i_16] * -0x1p+0f/*-1.000000e+00*/) * s11);
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 512)] = ((bfloat16_t)condval_1);
    float condval_2;
    if ((((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) < ((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)))) {
      condval_2 = ((G22[i_16] * -0x1p+0f/*-1.000000e+00*/) * s22);
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 1280)] = ((bfloat16_t)condval_2);
    float condval_3;
    if ((((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) < ((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)))) {
      condval_3 = ((G33[i_16] * -0x1p+0f/*-1.000000e+00*/) * s33);
    } else {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 2304)] = ((bfloat16_t)condval_3);
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 256)] = ((bfloat16_t)(G10[i_16] * s10));
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 768)] = ((bfloat16_t)(G20[i_16] * s20));
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 1024)] = ((bfloat16_t)(G21[i_16] * s21));
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 1536)] = ((bfloat16_t)(G30[i_16] * s30));
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 1792)] = ((bfloat16_t)(G31[i_16] * s31));
    ((bfloat16_t*)a_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 2048)] = ((bfloat16_t)(G32[i_16] * s32));
    float condval_4;
    if ((((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) == ((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)))) {
      condval_4 = 0x1p+0f/*1.000000e+00*/;
    } else {
      condval_4 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)i_s)[(((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1))] = ((bfloat16_t)condval_4);
    float condval_5;
    if ((((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) == ((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)))) {
      condval_5 = 0x1p+0f/*1.000000e+00*/;
    } else {
      condval_5 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)i_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 256)] = ((bfloat16_t)condval_5);
    float condval_6;
    if ((((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) == ((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)))) {
      condval_6 = 0x1p+0f/*1.000000e+00*/;
    } else {
      condval_6 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)i_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 512)] = ((bfloat16_t)condval_6);
    float condval_7;
    if ((((((i_16 & 3) >> 1) * 8) + (((int)threadIdx.x) >> 2)) == ((((i_16 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)))) {
      condval_7 = 0x1p+0f/*1.000000e+00*/;
    } else {
      condval_7 = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)i_s)[((((((((i_16 & 3) >> 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_16 >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_16 & 1)) + 768)] = ((bfloat16_t)condval_7);
  }
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_17 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
  }
  {
    bfloat16_t A_local_10[8];
    bfloat16_t B_local_10[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_10[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_10[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_10 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 4));
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 4; ++i_18) {
    *(uint1*)(i_s_local_cast + 0) = *(uint1*)(((bfloat16_t*)i_s) + (((((i_18 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_18 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    uint1 __18;
    float2 __19;
      float2 __20;
      uint1 v__9 = *(uint1*)(i_s_local_cast + 0);
      ((float2*)(&__20))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
      float2 v__10 = *(float2*)(tmp + (i_18 * 2));
      __19.x = (__20.x+v__10.x);
      __19.y = (__20.y+v__10.y);
    (reinterpret_cast<__nv_bfloat162*>(&__18))[0] = __float22bfloat162_rn(((float2*)(&__19))[0]);
    *(uint1*)(i_s_local_cast + 0) = __18;
    __syncthreads();
    *(uint1*)(((bfloat16_t*)i_s) + (((((i_18 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_18 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(i_s_local_cast + 0);
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    float broadcast_var_15 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_19 * 4)) = make_float4(broadcast_var_15, broadcast_var_15, broadcast_var_15, broadcast_var_15);
  }
  {
    bfloat16_t A_local_11[8];
    bfloat16_t B_local_11[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_11[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_11[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_11 + 0), reinterpret_cast<const unsigned*>(B_local_11 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 4; ++i_20) {
    uint1 __21;
    float2 v__11 = *(float2*)(tmp + (i_20 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__21))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
    *(uint1*)(a_s_local_cast_1 + 0) = __21;
    *(uint1*)(((bfloat16_t*)a_s) + (((((i_20 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(a_s_local_cast_1 + 0);
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_21 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
  }
  {
    bfloat16_t A_local_12[8];
    bfloat16_t B_local_12[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(A_local_12[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(B_local_12[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_12 + 0), reinterpret_cast<const unsigned*>(B_local_12 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_12 + 0), reinterpret_cast<const unsigned*>(B_local_12 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_22 = 0; i_22 < 4; ++i_22) {
    *(uint1*)(i_s_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)i_s) + ((((((i_22 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_22 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 256));
    uint1 __22;
    float2 __23;
      float2 __24;
      uint1 v__12 = *(uint1*)(i_s_local_cast_2 + 0);
      ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[0]);
      float2 v__13 = *(float2*)(tmp + (i_22 * 2));
      __23.x = (__24.x+v__13.x);
      __23.y = (__24.y+v__13.y);
    (reinterpret_cast<__nv_bfloat162*>(&__22))[0] = __float22bfloat162_rn(((float2*)(&__23))[0]);
    *(uint1*)(i_s_local_cast_2 + 0) = __22;
    *(uint1*)(((bfloat16_t*)i_s) + ((((((i_22 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_22 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 256)) = *(uint1*)(i_s_local_cast_2 + 0);
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 2; ++i_23) {
    float broadcast_var_17 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_23 * 4)) = make_float4(broadcast_var_17, broadcast_var_17, broadcast_var_17, broadcast_var_17);
  }
  {
    bfloat16_t A_local_13[8];
    bfloat16_t B_local_13[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(A_local_13[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(B_local_13[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_13 + 0), reinterpret_cast<const unsigned*>(B_local_13 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_13 + 0), reinterpret_cast<const unsigned*>(B_local_13 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 4; ++i_24) {
    uint1 __25;
    float2 v__14 = *(float2*)(tmp + (i_24 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__25))[0] = __float22bfloat162_rn(((float2*)(&v__14))[0]);
    *(uint1*)(a_s_local_cast_3 + 0) = __25;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_24 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_24 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 512)) = *(uint1*)(a_s_local_cast_3 + 0);
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_25 * 4)) = make_float4(broadcast_var_18, broadcast_var_18, broadcast_var_18, broadcast_var_18);
  }
  {
    bfloat16_t A_local_14[8];
    bfloat16_t B_local_14[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1280)])), (&(A_local_14[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(B_local_14[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_14 + 0), reinterpret_cast<const unsigned*>(B_local_14 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 4; ++i_26) {
    *(uint1*)(i_s_local_cast_4 + 0) = *(uint1*)(((bfloat16_t*)i_s) + ((((((i_26 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_26 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 512));
    uint1 __26;
    float2 __27;
      float2 __28;
      uint1 v__15 = *(uint1*)(i_s_local_cast_4 + 0);
      ((float2*)(&__28))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__15))[0]);
      float2 v__16 = *(float2*)(tmp + (i_26 * 2));
      __27.x = (__28.x+v__16.x);
      __27.y = (__28.y+v__16.y);
    (reinterpret_cast<__nv_bfloat162*>(&__26))[0] = __float22bfloat162_rn(((float2*)(&__27))[0]);
    *(uint1*)(i_s_local_cast_4 + 0) = __26;
    *(uint1*)(((bfloat16_t*)i_s) + ((((((i_26 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_26 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 512)) = *(uint1*)(i_s_local_cast_4 + 0);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    float broadcast_var_19 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_27 * 4)) = make_float4(broadcast_var_19, broadcast_var_19, broadcast_var_19, broadcast_var_19);
  }
  {
    bfloat16_t A_local_15[8];
    bfloat16_t B_local_15[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1280)])), (&(A_local_15[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1280)])), (&(B_local_15[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_15 + 0), reinterpret_cast<const unsigned*>(B_local_15 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 4; ++i_28) {
    uint1 __29;
    float2 v__17 = *(float2*)(tmp + (i_28 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__29))[0] = __float22bfloat162_rn(((float2*)(&v__17))[0]);
    *(uint1*)(a_s_local_cast_5 + 0) = __29;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_28 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_28 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1280)) = *(uint1*)(a_s_local_cast_5 + 0);
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 2; ++i_29) {
    float broadcast_var_20 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_29 * 4)) = make_float4(broadcast_var_20, broadcast_var_20, broadcast_var_20, broadcast_var_20);
  }
  {
    bfloat16_t A_local_16[8];
    bfloat16_t B_local_16[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 2304)])), (&(A_local_16[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 768)])), (&(B_local_16[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_16 + 0), reinterpret_cast<const unsigned*>(B_local_16 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 4; ++i_30) {
    *(uint1*)(i_s_local_cast_6 + 0) = *(uint1*)(((bfloat16_t*)i_s) + ((((((i_30 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_30 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 768));
    uint1 __30;
    float2 __31;
      float2 __32;
      uint1 v__18 = *(uint1*)(i_s_local_cast_6 + 0);
      ((float2*)(&__32))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__18))[0]);
      float2 v__19 = *(float2*)(tmp + (i_30 * 2));
      __31.x = (__32.x+v__19.x);
      __31.y = (__32.y+v__19.y);
    (reinterpret_cast<__nv_bfloat162*>(&__30))[0] = __float22bfloat162_rn(((float2*)(&__31))[0]);
    *(uint1*)(i_s_local_cast_6 + 0) = __30;
    *(uint1*)(((bfloat16_t*)i_s) + ((((((i_30 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_30 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 768)) = *(uint1*)(i_s_local_cast_6 + 0);
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 2; ++i_31) {
    float broadcast_var_21 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_31 * 4)) = make_float4(broadcast_var_21, broadcast_var_21, broadcast_var_21, broadcast_var_21);
  }
  {
    bfloat16_t A_local_17[8];
    bfloat16_t B_local_17[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 2304)])), (&(A_local_17[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 2304)])), (&(B_local_17[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_17 + 0), reinterpret_cast<const unsigned*>(B_local_17 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_32 = 0; i_32 < 4; ++i_32) {
    uint1 __33;
    float2 v__20 = *(float2*)(tmp + (i_32 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__33))[0] = __float22bfloat162_rn(((float2*)(&v__20))[0]);
    *(uint1*)(a_s_local_cast_7 + 0) = __33;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_32 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_32 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2304)) = *(uint1*)(a_s_local_cast_7 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 2; ++i_33) {
    float broadcast_var_22 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_33 * 4)) = make_float4(broadcast_var_22, broadcast_var_22, broadcast_var_22, broadcast_var_22);
  }
  {
    bfloat16_t A_local_18[8];
    bfloat16_t B_local_18[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(A_local_18[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(B_local_18[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_18 + 0), reinterpret_cast<const unsigned*>(B_local_18 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_34 = 0; i_34 < 4; ++i_34) {
    uint1 __34;
    float2 v__21 = *(float2*)(tmp + (i_34 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__34))[0] = __float22bfloat162_rn(((float2*)(&v__21))[0]);
    *(uint1*)(work_s_local_cast_8 + 0) = __34;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_34 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_34 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_8 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 2; ++i_35) {
    float broadcast_var_23 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_35 * 4)) = make_float4(broadcast_var_23, broadcast_var_23, broadcast_var_23, broadcast_var_23);
  }
  {
    bfloat16_t A_local_19[8];
    bfloat16_t B_local_19[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)work_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_19[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_19[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_19 + 0), reinterpret_cast<const unsigned*>(B_local_19 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_36 = 0; i_36 < 4; ++i_36) {
    float broadcast_var_24 = -0x1p+0f/*-1.000000e+00*/;
    uint1 __35;
    float2 __36;
      float2 v__22 = *(float2*)(tmp + (i_36 * 2));
      float2 v__23 = make_float2(broadcast_var_24, broadcast_var_24);
      __36.x = (v__22.x*v__23.x);
      __36.y = (v__22.y*v__23.y);
    (reinterpret_cast<__nv_bfloat162*>(&__35))[0] = __float22bfloat162_rn(((float2*)(&__36))[0]);
    *(uint1*)(a_s_local_cast_9 + 0) = __35;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_36 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_36 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 256)) = *(uint1*)(a_s_local_cast_9 + 0);
  }
  #pragma unroll
  for (int i_37 = 0; i_37 < 2; ++i_37) {
    float broadcast_var_25 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_37 * 4)) = make_float4(broadcast_var_25, broadcast_var_25, broadcast_var_25, broadcast_var_25);
  }
  {
    bfloat16_t A_local_20[8];
    bfloat16_t B_local_20[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(A_local_20[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1024)])), (&(B_local_20[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_20 + 0), reinterpret_cast<const unsigned*>(B_local_20 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_20 + 0), reinterpret_cast<const unsigned*>(B_local_20 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_38 = 0; i_38 < 4; ++i_38) {
    uint1 __37;
    float2 v__24 = *(float2*)(tmp + (i_38 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__37))[0] = __float22bfloat162_rn(((float2*)(&v__24))[0]);
    *(uint1*)(work_s_local_cast_10 + 0) = __37;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_38 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_38 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_10 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_39 = 0; i_39 < 2; ++i_39) {
    float broadcast_var_26 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_39 * 4)) = make_float4(broadcast_var_26, broadcast_var_26, broadcast_var_26, broadcast_var_26);
  }
  {
    bfloat16_t A_local_21[8];
    bfloat16_t B_local_21[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)work_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_21[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(B_local_21[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_21 + 0), reinterpret_cast<const unsigned*>(B_local_21 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_21 + 0), reinterpret_cast<const unsigned*>(B_local_21 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    float broadcast_var_27 = -0x1p+0f/*-1.000000e+00*/;
    uint1 __38;
    float2 __39;
      float2 v__25 = *(float2*)(tmp + (i_40 * 2));
      float2 v__26 = make_float2(broadcast_var_27, broadcast_var_27);
      __39.x = (v__25.x*v__26.x);
      __39.y = (v__25.y*v__26.y);
    (reinterpret_cast<__nv_bfloat162*>(&__38))[0] = __float22bfloat162_rn(((float2*)(&__39))[0]);
    *(uint1*)(a_s_local_cast_11 + 0) = __38;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_40 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_40 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)) = *(uint1*)(a_s_local_cast_11 + 0);
  }
  #pragma unroll
  for (int i_41 = 0; i_41 < 2; ++i_41) {
    float broadcast_var_28 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_41 * 4)) = make_float4(broadcast_var_28, broadcast_var_28, broadcast_var_28, broadcast_var_28);
  }
  {
    bfloat16_t A_local_22[8];
    bfloat16_t B_local_22[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 768)])), (&(A_local_22[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_22[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_22 + 0), reinterpret_cast<const unsigned*>(B_local_22 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_42 = 0; i_42 < 4; ++i_42) {
    uint1 __40;
    float2 v__27 = *(float2*)(tmp + (i_42 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__40))[0] = __float22bfloat162_rn(((float2*)(&v__27))[0]);
    *(uint1*)(work_s_local_cast_12 + 0) = __40;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_42 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_42 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_12 + 0);
  }
  #pragma unroll
  for (int i_43 = 0; i_43 < 2; ++i_43) {
    float broadcast_var_29 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_43 * 4)) = make_float4(broadcast_var_29, broadcast_var_29, broadcast_var_29, broadcast_var_29);
  }
  {
    bfloat16_t A_local_23[8];
    bfloat16_t B_local_23[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1024)])), (&(A_local_23[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(B_local_23[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_23 + 0), reinterpret_cast<const unsigned*>(B_local_23 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_44 = 0; i_44 < 4; ++i_44) {
    *(uint1*)(work_s_local_cast_13 + 0) = *(uint1*)(((bfloat16_t*)work_s) + (((((i_44 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_44 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    uint1 __41;
    float2 __42;
      float2 __43;
      uint1 v__28 = *(uint1*)(work_s_local_cast_13 + 0);
      ((float2*)(&__43))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__28))[0]);
      float2 v__29 = *(float2*)(tmp + (i_44 * 2));
      __42.x = (__43.x+v__29.x);
      __42.y = (__43.y+v__29.y);
    (reinterpret_cast<__nv_bfloat162*>(&__41))[0] = __float22bfloat162_rn(((float2*)(&__42))[0]);
    *(uint1*)(work_s_local_cast_13 + 0) = __41;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_44 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_44 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_13 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_45 = 0; i_45 < 2; ++i_45) {
    float broadcast_var_30 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_45 * 4)) = make_float4(broadcast_var_30, broadcast_var_30, broadcast_var_30, broadcast_var_30);
  }
  {
    bfloat16_t A_local_24[8];
    bfloat16_t B_local_24[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(A_local_24[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)work_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_24[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_24 + 0), reinterpret_cast<const unsigned*>(B_local_24 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_24 + 0), reinterpret_cast<const unsigned*>(B_local_24 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 4; ++i_46) {
    float broadcast_var_31 = -0x1p+0f/*-1.000000e+00*/;
    uint1 __44;
    float2 __45;
      float2 v__30 = *(float2*)(tmp + (i_46 * 2));
      float2 v__31 = make_float2(broadcast_var_31, broadcast_var_31);
      __45.x = (v__30.x*v__31.x);
      __45.y = (v__30.y*v__31.y);
    (reinterpret_cast<__nv_bfloat162*>(&__44))[0] = __float22bfloat162_rn(((float2*)(&__45))[0]);
    *(uint1*)(a_s_local_cast_14 + 0) = __44;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_46 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_46 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 768)) = *(uint1*)(a_s_local_cast_14 + 0);
  }
  #pragma unroll
  for (int i_47 = 0; i_47 < 2; ++i_47) {
    float broadcast_var_32 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_47 * 4)) = make_float4(broadcast_var_32, broadcast_var_32, broadcast_var_32, broadcast_var_32);
  }
  {
    bfloat16_t A_local_25[8];
    bfloat16_t B_local_25[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1536)])), (&(A_local_25[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_25[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_25 + 0), reinterpret_cast<const unsigned*>(B_local_25 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_25 + 0), reinterpret_cast<const unsigned*>(B_local_25 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_48 = 0; i_48 < 4; ++i_48) {
    uint1 __46;
    float2 v__32 = *(float2*)(tmp + (i_48 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__46))[0] = __float22bfloat162_rn(((float2*)(&v__32))[0]);
    *(uint1*)(work_s_local_cast_15 + 0) = __46;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_48 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_48 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_15 + 0);
  }
  #pragma unroll
  for (int i_49 = 0; i_49 < 2; ++i_49) {
    float broadcast_var_33 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_49 * 4)) = make_float4(broadcast_var_33, broadcast_var_33, broadcast_var_33, broadcast_var_33);
  }
  {
    bfloat16_t A_local_26[8];
    bfloat16_t B_local_26[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1792)])), (&(A_local_26[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(B_local_26[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_26 + 0), reinterpret_cast<const unsigned*>(B_local_26 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_26 + 0), reinterpret_cast<const unsigned*>(B_local_26 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_50 = 0; i_50 < 4; ++i_50) {
    *(uint1*)(work_s_local_cast_16 + 0) = *(uint1*)(((bfloat16_t*)work_s) + (((((i_50 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_50 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    uint1 __47;
    float2 __48;
      float2 __49;
      uint1 v__33 = *(uint1*)(work_s_local_cast_16 + 0);
      ((float2*)(&__49))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__33))[0]);
      float2 v__34 = *(float2*)(tmp + (i_50 * 2));
      __48.x = (__49.x+v__34.x);
      __48.y = (__49.y+v__34.y);
    (reinterpret_cast<__nv_bfloat162*>(&__47))[0] = __float22bfloat162_rn(((float2*)(&__48))[0]);
    *(uint1*)(work_s_local_cast_16 + 0) = __47;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_50 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_50 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_16 + 0);
  }
  #pragma unroll
  for (int i_51 = 0; i_51 < 2; ++i_51) {
    float broadcast_var_34 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_51 * 4)) = make_float4(broadcast_var_34, broadcast_var_34, broadcast_var_34, broadcast_var_34);
  }
  {
    bfloat16_t A_local_27[8];
    bfloat16_t B_local_27[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 2048)])), (&(A_local_27[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 768)])), (&(B_local_27[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_27 + 0), reinterpret_cast<const unsigned*>(B_local_27 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_27 + 0), reinterpret_cast<const unsigned*>(B_local_27 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_52 = 0; i_52 < 4; ++i_52) {
    *(uint1*)(work_s_local_cast_17 + 0) = *(uint1*)(((bfloat16_t*)work_s) + (((((i_52 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_52 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    uint1 __50;
    float2 __51;
      float2 __52;
      uint1 v__35 = *(uint1*)(work_s_local_cast_17 + 0);
      ((float2*)(&__52))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__35))[0]);
      float2 v__36 = *(float2*)(tmp + (i_52 * 2));
      __51.x = (__52.x+v__36.x);
      __51.y = (__52.y+v__36.y);
    (reinterpret_cast<__nv_bfloat162*>(&__50))[0] = __float22bfloat162_rn(((float2*)(&__51))[0]);
    *(uint1*)(work_s_local_cast_17 + 0) = __50;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_52 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_52 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_17 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_53 = 0; i_53 < 2; ++i_53) {
    float broadcast_var_35 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_53 * 4)) = make_float4(broadcast_var_35, broadcast_var_35, broadcast_var_35, broadcast_var_35);
  }
  {
    bfloat16_t A_local_28[8];
    bfloat16_t B_local_28[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 768)])), (&(A_local_28[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)work_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_28[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_28 + 0), reinterpret_cast<const unsigned*>(B_local_28 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_28 + 0), reinterpret_cast<const unsigned*>(B_local_28 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_54 = 0; i_54 < 4; ++i_54) {
    float broadcast_var_36 = -0x1p+0f/*-1.000000e+00*/;
    uint1 __53;
    float2 __54;
      float2 v__37 = *(float2*)(tmp + (i_54 * 2));
      float2 v__38 = make_float2(broadcast_var_36, broadcast_var_36);
      __54.x = (v__37.x*v__38.x);
      __54.y = (v__37.y*v__38.y);
    (reinterpret_cast<__nv_bfloat162*>(&__53))[0] = __float22bfloat162_rn(((float2*)(&__54))[0]);
    *(uint1*)(a_s_local_cast_18 + 0) = __53;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_54 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_54 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1536)) = *(uint1*)(a_s_local_cast_18 + 0);
  }
  #pragma unroll
  for (int i_55 = 0; i_55 < 2; ++i_55) {
    float broadcast_var_37 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_55 * 4)) = make_float4(broadcast_var_37, broadcast_var_37, broadcast_var_37, broadcast_var_37);
  }
  {
    bfloat16_t A_local_29[8];
    bfloat16_t B_local_29[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1792)])), (&(A_local_29[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 256)])), (&(B_local_29[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_29 + 0), reinterpret_cast<const unsigned*>(B_local_29 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_29 + 0), reinterpret_cast<const unsigned*>(B_local_29 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_56 = 0; i_56 < 4; ++i_56) {
    uint1 __55;
    float2 v__39 = *(float2*)(tmp + (i_56 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__55))[0] = __float22bfloat162_rn(((float2*)(&v__39))[0]);
    *(uint1*)(work_s_local_cast_19 + 0) = __55;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_56 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_56 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_19 + 0);
  }
  #pragma unroll
  for (int i_57 = 0; i_57 < 2; ++i_57) {
    float broadcast_var_38 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_57 * 4)) = make_float4(broadcast_var_38, broadcast_var_38, broadcast_var_38, broadcast_var_38);
  }
  {
    bfloat16_t A_local_30[8];
    bfloat16_t B_local_30[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 2048)])), (&(A_local_30[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 1024)])), (&(B_local_30[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_30 + 0), reinterpret_cast<const unsigned*>(B_local_30 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_30 + 0), reinterpret_cast<const unsigned*>(B_local_30 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_58 = 0; i_58 < 4; ++i_58) {
    *(uint1*)(work_s_local_cast_20 + 0) = *(uint1*)(((bfloat16_t*)work_s) + (((((i_58 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_58 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    uint1 __56;
    float2 __57;
      float2 __58;
      uint1 v__40 = *(uint1*)(work_s_local_cast_20 + 0);
      ((float2*)(&__58))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__40))[0]);
      float2 v__41 = *(float2*)(tmp + (i_58 * 2));
      __57.x = (__58.x+v__41.x);
      __57.y = (__58.y+v__41.y);
    (reinterpret_cast<__nv_bfloat162*>(&__56))[0] = __float22bfloat162_rn(((float2*)(&__57))[0]);
    *(uint1*)(work_s_local_cast_20 + 0) = __56;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_58 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_58 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_20 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_59 = 0; i_59 < 2; ++i_59) {
    float broadcast_var_39 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_59 * 4)) = make_float4(broadcast_var_39, broadcast_var_39, broadcast_var_39, broadcast_var_39);
  }
  {
    bfloat16_t A_local_31[8];
    bfloat16_t B_local_31[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 768)])), (&(A_local_31[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)work_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_31[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_31 + 0), reinterpret_cast<const unsigned*>(B_local_31 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_31 + 0), reinterpret_cast<const unsigned*>(B_local_31 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_60 = 0; i_60 < 4; ++i_60) {
    float broadcast_var_40 = -0x1p+0f/*-1.000000e+00*/;
    uint1 __59;
    float2 __60;
      float2 v__42 = *(float2*)(tmp + (i_60 * 2));
      float2 v__43 = make_float2(broadcast_var_40, broadcast_var_40);
      __60.x = (v__42.x*v__43.x);
      __60.y = (v__42.y*v__43.y);
    (reinterpret_cast<__nv_bfloat162*>(&__59))[0] = __float22bfloat162_rn(((float2*)(&__60))[0]);
    *(uint1*)(a_s_local_cast_21 + 0) = __59;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_60 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_60 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1792)) = *(uint1*)(a_s_local_cast_21 + 0);
  }
  #pragma unroll
  for (int i_61 = 0; i_61 < 2; ++i_61) {
    float broadcast_var_41 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_61 * 4)) = make_float4(broadcast_var_41, broadcast_var_41, broadcast_var_41, broadcast_var_41);
  }
  {
    bfloat16_t A_local_32[8];
    bfloat16_t B_local_32[8];
    __syncthreads();
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 768)])), (&(A_local_32[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)a_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 2048)])), (&(B_local_32[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_32 + 0), reinterpret_cast<const unsigned*>(B_local_32 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_32 + 0), reinterpret_cast<const unsigned*>(B_local_32 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_62 = 0; i_62 < 4; ++i_62) {
    uint1 __61;
    float2 v__44 = *(float2*)(tmp + (i_62 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__61))[0] = __float22bfloat162_rn(((float2*)(&v__44))[0]);
    *(uint1*)(work_s_local_cast_22 + 0) = __61;
    *(uint1*)(((bfloat16_t*)work_s) + (((((i_62 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_62 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(work_s_local_cast_22 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_63 = 0; i_63 < 2; ++i_63) {
    float broadcast_var_42 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(tmp + (i_63 * 4)) = make_float4(broadcast_var_42, broadcast_var_42, broadcast_var_42, broadcast_var_42);
  }
  {
    bfloat16_t A_local_33[8];
    bfloat16_t B_local_33[8];
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)work_s)[(((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_33[0])));
    tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)i_s)[((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 512)])), (&(B_local_33[0])));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 0), reinterpret_cast<const unsigned*>(A_local_33 + 0), reinterpret_cast<const unsigned*>(B_local_33 + 0));
    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(tmp + 4), reinterpret_cast<const unsigned*>(A_local_33 + 0), reinterpret_cast<const unsigned*>(B_local_33 + 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_64 = 0; i_64 < 4; ++i_64) {
    float broadcast_var_43 = -0x1p+0f/*-1.000000e+00*/;
    uint1 __62;
    float2 __63;
      float2 v__45 = *(float2*)(tmp + (i_64 * 2));
      float2 v__46 = make_float2(broadcast_var_43, broadcast_var_43);
      __63.x = (v__45.x*v__46.x);
      __63.y = (v__45.y*v__46.y);
    (reinterpret_cast<__nv_bfloat162*>(&__62))[0] = __float22bfloat162_rn(((float2*)(&__63))[0]);
    *(uint1*)(a_s_local_cast_23 + 0) = __62;
    *(uint1*)(((bfloat16_t*)a_s) + ((((((i_64 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((((((int)threadIdx.x) >> 4) + (i_64 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 2048)) = *(uint1*)(a_s_local_cast_23 + 0);
  }
  __syncthreads();
  *(uint4*)(A + ((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(((bfloat16_t*)i_s) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 16384)) = *(uint4*)(((bfloat16_t*)a_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 256));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 16400)) = *(uint4*)(((bfloat16_t*)i_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 256));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 32768)) = *(uint4*)(((bfloat16_t*)a_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 768));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 32784)) = *(uint4*)(((bfloat16_t*)a_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1024));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 32800)) = *(uint4*)(((bfloat16_t*)i_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 49152)) = *(uint4*)(((bfloat16_t*)a_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1536));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 49168)) = *(uint4*)(((bfloat16_t*)a_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1792));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 49184)) = *(uint4*)(((bfloat16_t*)a_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048));
  *(uint4*)(A + (((((((int)blockIdx.z) * 65536) + ((((int)threadIdx.x) >> 1) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 49200)) = *(uint4*)(((bfloat16_t*)i_s) + ((((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 768));
}
