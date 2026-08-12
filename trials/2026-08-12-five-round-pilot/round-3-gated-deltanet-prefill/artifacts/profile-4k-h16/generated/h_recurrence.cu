#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
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

extern "C" __global__ void h_recurrence_bthd_kernel(bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ S_0, const bfloat16_t* __restrict__ g, const bfloat16_t* __restrict__ k, const bfloat16_t* __restrict__ u, bfloat16_t* __restrict__ v_new, const bfloat16_t* __restrict__ w);
extern "C" __global__ void __launch_bounds__(128, 1) h_recurrence_bthd_kernel(bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ S_0, const bfloat16_t* __restrict__ g, const bfloat16_t* __restrict__ k, const bfloat16_t* __restrict__ u, bfloat16_t* __restrict__ v_new, const bfloat16_t* __restrict__ w) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_c = ((void*)((char*)buf_dyn_shmem + 0));
  void* u_c = ((void*)((char*)buf_dyn_shmem + 4096));
  void* g_c = ((void*)((char*)buf_dyn_shmem + 8192));
  void* k_c = ((void*)((char*)buf_dyn_shmem + 9216));
  void* w_c = ((void*)((char*)buf_dyn_shmem + 41984));
  void* v_new_c = ((void*)((char*)buf_dyn_shmem + 74752));
  float ws_frag[8];
  float h_next_frag[16];
  bfloat16_t u_c_local_cast_1[2];
  bfloat16_t g_c_local_cast_2[2];
  bfloat16_t v_new_c_local_cast[2];
  bfloat16_t v_new_c_local_cast_3[8];
  bfloat16_t h_c_local_cast_4[2];
  bfloat16_t u_c_local_cast_6[2];
  bfloat16_t g_c_local_cast_7[2];
  bfloat16_t v_new_c_local_cast_5[2];
  bfloat16_t v_new_c_local_cast_8[8];
  bfloat16_t h_c_local_cast_9[2];
  bfloat16_t u_c_local_cast_11[2];
  bfloat16_t g_c_local_cast_12[2];
  bfloat16_t v_new_c_local_cast_10[2];
  bfloat16_t v_new_c_local_cast_13[8];
  bfloat16_t h_c_local_cast_14[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(((bfloat16_t*)h_c) + (((i * 1024) + ((((int)threadIdx.x) >> 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(S_0 + (((((((int)blockIdx.z) * 16384) + (i * 8192)) + ((((int)threadIdx.x) >> 1) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(S + (((((((int)blockIdx.z) * 1064960) + (i_1 * 8192)) + ((((int)threadIdx.x) >> 1) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(((bfloat16_t*)h_c) + (((i_1 * 1024) + ((((int)threadIdx.x) >> 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  }
  __syncthreads();
  tl::cp_async_gs<16>((&(((bfloat16_t*)u_c)[(((int)threadIdx.x) * 8)])), (&(u[(((((((int)threadIdx.x) >> 1) * 2048) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8))])));
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((bfloat16_t*)g_c)[((int)threadIdx.x)] = g[((((int)threadIdx.x) * 16) + ((int)blockIdx.z))];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)k_c)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[((((i_2 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
    tl::cp_async_gs<16>((&(((bfloat16_t*)w_c)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(w[((((i_2 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((bfloat16_t*)u_c)[((((int)threadIdx.x) * 8) + 1024)])), (&(u[((((((((int)threadIdx.x) >> 1) * 2048) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 131072)])));
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((bfloat16_t*)g_c)[(((int)threadIdx.x) + 64)] = g[(((((int)threadIdx.x) * 16) + ((int)blockIdx.z)) + 1024)];
  }
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)k_c)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(k[(((((i_3 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
    tl::cp_async_gs<16>((&(((bfloat16_t*)w_c)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(w[(((((i_3 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
  }
  tl::cp_async_commit();
  for (int t = 0; t < 62; ++t) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(ws_frag + (i_4 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)w_c)[((t & 1) * 8192)])));
      tl::initialize_wgmma_descriptor<3, 0, 16>(desc_b, (&(((bfloat16_t*)h_c)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(ws_frag + 0), 8);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 16, 16, false, true, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((ki * 512) >> 4)), ((uint32_t*)(ws_frag + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(ws_frag + 0), 8);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(uint1*)(u_c_local_cast_1 + 0) = *(uint1*)(((bfloat16_t*)u_c) + (((((((t & 1) * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_5 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      *(uint1*)(g_c_local_cast_2 + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[(((((t & 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((bfloat16_t*)g_c)[(((((t & 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
      float broadcast_var_1 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
      uint1 __1;
      float2 __2;
        float2 __3;
        uint1 v_ = *(uint1*)(u_c_local_cast_1 + 0);
        ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        float2 __4;
          float2 v__1 = *(float2*)(ws_frag + (i_5 * 2));
          float2 __5;
          float2 __6;
            float2 __7;
            uint1 __8;
              uint1 v__2 = *(uint1*)(g_c_local_cast_2 + 0);
              uint1 v__3 = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)], ((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)]));
              *(uint1*)(&(__8.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__3.x)))));
            ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__8))[0]);
            float2 v__4 = make_float2(broadcast_var_1, broadcast_var_1);
            __6.x = (__7.x*v__4.x);
            __6.y = (__7.y*v__4.y);
          __5.x = exp2f(__6.x);
          __5.y = exp2f(__6.y);
          __4.x = (v__1.x*__5.x);
          __4.y = (v__1.y*__5.y);
        __2.x = (__3.x-__4.x);
        __2.y = (__3.y-__4.y);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      *(uint1*)(v_new_c_local_cast + 0) = __1;
      *(uint1*)(((bfloat16_t*)v_new_c) + ((((((((int)threadIdx.x) >> 5) * 256) + ((i_5 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_5 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(v_new_c_local_cast + 0);
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((bfloat16_t*)u_c)[(((t & 1) * 1024) + (((int)threadIdx.x) * 8))])), (&(u[((((((t * 131072) + ((((int)threadIdx.x) >> 1) * 2048)) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 262144)])));
    tl::cp_async_commit();
    __syncthreads();
    *(uint4*)(v_new + (((((t * 131072) + ((((int)threadIdx.x) >> 1) * 2048)) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    *(uint4*)(v_new_c_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      uint2 __9;
      float4 __10;
        float4 __11;
        uint2 v__5 = *(uint2*)(v_new_c_local_cast_3 + (i_6 * 4));
        ((float2*)(&__11))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
        ((float2*)(&__11))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[1]);
        float4 v__6 = make_float4(exp2f((((float)(((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)] - ((bfloat16_t*)g_c)[(((t & 1) * 64) + (((int)threadIdx.x) >> 1))])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)] - ((bfloat16_t*)g_c)[(((t & 1) * 64) + (((int)threadIdx.x) >> 1))])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)] - ((bfloat16_t*)g_c)[(((t & 1) * 64) + (((int)threadIdx.x) >> 1))])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)] - ((bfloat16_t*)g_c)[(((t & 1) * 64) + (((int)threadIdx.x) >> 1))])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
        __10.x = (__11.x*v__6.x);
        __10.y = (__11.y*v__6.y);
        __10.z = (__11.z*v__6.z);
        __10.w = (__11.w*v__6.w);
      (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&__10))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__9))[1] = __float22bfloat162_rn(((float2*)(&__10))[1]);
      *(uint2*)(v_new_c_local_cast_3 + (i_6 * 4)) = __9;
    }
    *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(v_new_c_local_cast_3 + 0);
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      *(uint1*)(h_c_local_cast_4 + 0) = *(uint1*)(((bfloat16_t*)h_c) + (((((((i_7 >> 2) * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_7 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __12;
        float2 __13;
        uint1 v__7 = *(uint1*)(h_c_local_cast_4 + 0);
        ((float2*)(&__13))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__7))[0]);
        float2 v__8 = make_float2(exp2f((((float)((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)((bfloat16_t*)g_c)[(((t & 1) * 64) + 63)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
        __12.x = (__13.x*v__8.x);
        __12.y = (__13.y*v__8.y);
      *(float2*)(h_next_frag + (i_7 * 2)) = __12;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((bfloat16_t*)g_c)[(((t & 1) * 64) + ((int)threadIdx.x))] = g[((((t * 1024) + (((int)threadIdx.x) * 16)) + ((int)blockIdx.z)) + 2048)];
    }
    tl::cp_async_commit();
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a_1, (&(((bfloat16_t*)k_c)[((t & 1) * 8192)])));
      tl::initialize_wgmma_descriptor<3, 0, 16>(desc_b_1, (&(((bfloat16_t*)v_new_c)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_next_frag + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 16, 16, true, true, 1, 1>(uint64_t(desc_a_1 + (((i_8 * 8192) + (ki_1 * 2048)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 512) >> 4)), ((uint32_t*)(h_next_frag + (i_8 * 8))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_next_frag + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)k_c)[((((((((t & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[((((((t * 131072) + (i_9 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144)])));
      tl::cp_async_gs<16>((&(((bfloat16_t*)w_c)[((((((((t & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(w[((((((t * 131072) + (i_9 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_c)[((((i_10 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), __pack_half2(((bfloat16_t)h_next_frag[(i_10 * 8)]), ((bfloat16_t)h_next_frag[((i_10 * 8) + 1)])), __pack_half2(((bfloat16_t)h_next_frag[((i_10 * 8) + 2)]), ((bfloat16_t)h_next_frag[((i_10 * 8) + 3)])), __pack_half2(((bfloat16_t)h_next_frag[((i_10 * 8) + 4)]), ((bfloat16_t)h_next_frag[((i_10 * 8) + 5)])), __pack_half2(((bfloat16_t)h_next_frag[((i_10 * 8) + 6)]), ((bfloat16_t)h_next_frag[((i_10 * 8) + 7)])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      *(uint4*)(S + (((((((((int)blockIdx.z) * 1064960) + (t * 16384)) + (i_11 * 8192)) + ((((int)threadIdx.x) >> 1) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 16384)) = *(uint4*)(((bfloat16_t*)h_c) + (((i_11 * 1024) + ((((int)threadIdx.x) >> 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 2; ++i_12) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ws_frag + (i_12 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_2;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)w_c)[0])));
    tl::initialize_wgmma_descriptor<3, 0, 16>(desc_b_2, (&(((bfloat16_t*)h_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(ws_frag + 0), 8);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 16, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 512) >> 4)), ((uint32_t*)(ws_frag + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(ws_frag + 0), 8);
  }
  __syncthreads();
  #pragma unroll
  for (int i_13 = 0; i_13 < 4; ++i_13) {
    *(uint1*)(u_c_local_cast_6 + 0) = *(uint1*)(((bfloat16_t*)u_c) + ((((((((int)threadIdx.x) >> 5) * 256) + ((i_13 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_13 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    *(uint1*)(g_c_local_cast_7 + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[((((((int)threadIdx.x) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((bfloat16_t*)g_c)[((((((int)threadIdx.x) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
    float broadcast_var_3 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
    uint1 __14;
    float2 __15;
      float2 __16;
      uint1 v__9 = *(uint1*)(u_c_local_cast_6 + 0);
      ((float2*)(&__16))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
      float2 __17;
        float2 v__10 = *(float2*)(ws_frag + (i_13 * 2));
        float2 __18;
        float2 __19;
          float2 __20;
          uint1 __21;
            uint1 v__11 = *(uint1*)(g_c_local_cast_7 + 0);
            uint1 v__12 = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[63], ((bfloat16_t*)g_c)[63]));
            *(uint1*)(&(__21.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__11.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__12.x)))));
          ((float2*)(&__20))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__21))[0]);
          float2 v__13 = make_float2(broadcast_var_3, broadcast_var_3);
          __19.x = (__20.x*v__13.x);
          __19.y = (__20.y*v__13.y);
        __18.x = exp2f(__19.x);
        __18.y = exp2f(__19.y);
        __17.x = (v__10.x*__18.x);
        __17.y = (v__10.y*__18.y);
      __15.x = (__16.x-__17.x);
      __15.y = (__16.y-__17.y);
    (reinterpret_cast<__nv_bfloat162*>(&__14))[0] = __float22bfloat162_rn(((float2*)(&__15))[0]);
    *(uint1*)(v_new_c_local_cast_5 + 0) = __14;
    *(uint1*)(((bfloat16_t*)v_new_c) + ((((((((int)threadIdx.x) >> 5) * 256) + ((i_13 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(v_new_c_local_cast_5 + 0);
  }
  __syncthreads();
  *(uint4*)(v_new + ((((((((int)threadIdx.x) >> 1) * 2048) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 8126464)) = *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  *(uint4*)(v_new_c_local_cast_8 + 0) = *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    uint2 __22;
    float4 __23;
      float4 __24;
      uint2 v__14 = *(uint2*)(v_new_c_local_cast_8 + (i_14 * 4));
      ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__14))[0]);
      ((float2*)(&__24))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__14))[1]);
      float4 v__15 = make_float4(exp2f((((float)(((bfloat16_t*)g_c)[63] - ((bfloat16_t*)g_c)[(((int)threadIdx.x) >> 1)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[63] - ((bfloat16_t*)g_c)[(((int)threadIdx.x) >> 1)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[63] - ((bfloat16_t*)g_c)[(((int)threadIdx.x) >> 1)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[63] - ((bfloat16_t*)g_c)[(((int)threadIdx.x) >> 1)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
      __23.x = (__24.x*v__15.x);
      __23.y = (__24.y*v__15.y);
      __23.z = (__24.z*v__15.z);
      __23.w = (__24.w*v__15.w);
    (reinterpret_cast<__nv_bfloat162*>(&__22))[0] = __float22bfloat162_rn(((float2*)(&__23))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__22))[1] = __float22bfloat162_rn(((float2*)(&__23))[1]);
    *(uint2*)(v_new_c_local_cast_8 + (i_14 * 4)) = __22;
  }
  *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(v_new_c_local_cast_8 + 0);
  #pragma unroll
  for (int i_15 = 0; i_15 < 8; ++i_15) {
    *(uint1*)(h_c_local_cast_9 + 0) = *(uint1*)(((bfloat16_t*)h_c) + (((((((i_15 >> 2) * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_15 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_15 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __25;
      float2 __26;
      uint1 v__16 = *(uint1*)(h_c_local_cast_9 + 0);
      ((float2*)(&__26))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__16))[0]);
      float2 v__17 = make_float2(exp2f((((float)((bfloat16_t*)g_c)[63]) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)((bfloat16_t*)g_c)[63]) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
      __25.x = (__26.x*v__17.x);
      __25.y = (__26.y*v__17.y);
    *(float2*)(h_next_frag + (i_15 * 2)) = __25;
  }
  {
    tl::GmmaDescriptor desc_a_3;
    tl::GmmaDescriptor desc_b_3;
    __syncthreads();
    tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a_3, (&(((bfloat16_t*)k_c)[0])));
    tl::initialize_wgmma_descriptor<3, 0, 16>(desc_b_3, (&(((bfloat16_t*)v_new_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_next_frag + 0), 16);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 16, 16, true, true, 1, 1>(uint64_t(desc_a_3 + (((i_16 * 8192) + (ki_3 * 2048)) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 512) >> 4)), ((uint32_t*)(h_next_frag + (i_16 * 8))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_next_frag + 0), 16);
  }
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_c)[((((i_17 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), __pack_half2(((bfloat16_t)h_next_frag[(i_17 * 8)]), ((bfloat16_t)h_next_frag[((i_17 * 8) + 1)])), __pack_half2(((bfloat16_t)h_next_frag[((i_17 * 8) + 2)]), ((bfloat16_t)h_next_frag[((i_17 * 8) + 3)])), __pack_half2(((bfloat16_t)h_next_frag[((i_17 * 8) + 4)]), ((bfloat16_t)h_next_frag[((i_17 * 8) + 5)])), __pack_half2(((bfloat16_t)h_next_frag[((i_17 * 8) + 6)]), ((bfloat16_t)h_next_frag[((i_17 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    *(uint4*)(S + ((((((((int)blockIdx.z) * 1064960) + (i_18 * 8192)) + ((((int)threadIdx.x) >> 1) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1032192)) = *(uint4*)(((bfloat16_t*)h_c) + (((i_18 * 1024) + ((((int)threadIdx.x) >> 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ws_frag + (i_19 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tl::GmmaDescriptor desc_a_4;
    tl::GmmaDescriptor desc_b_4;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((bfloat16_t*)w_c)[8192])));
    tl::initialize_wgmma_descriptor<3, 0, 16>(desc_b_4, (&(((bfloat16_t*)h_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(ws_frag + 0), 8);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 16, 16, false, true, 1, 1>(uint64_t(desc_a_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((ki_4 * 512) >> 4)), ((uint32_t*)(ws_frag + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(ws_frag + 0), 8);
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 4; ++i_20) {
    *(uint1*)(u_c_local_cast_11 + 0) = *(uint1*)(((bfloat16_t*)u_c) + (((((((((int)threadIdx.x) >> 5) * 256) + ((i_20 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_20 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024));
    *(uint1*)(g_c_local_cast_12 + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_20 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)], ((bfloat16_t*)g_c)[(((((((int)threadIdx.x) >> 5) * 16) + ((i_20 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)]));
    float broadcast_var_5 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
    uint1 __27;
    float2 __28;
      float2 __29;
      uint1 v__18 = *(uint1*)(u_c_local_cast_11 + 0);
      ((float2*)(&__29))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__18))[0]);
      float2 __30;
        float2 v__19 = *(float2*)(ws_frag + (i_20 * 2));
        float2 __31;
        float2 __32;
          float2 __33;
          uint1 __34;
            uint1 v__20 = *(uint1*)(g_c_local_cast_12 + 0);
            uint1 v__21 = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[127], ((bfloat16_t*)g_c)[127]));
            *(uint1*)(&(__34.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__20.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__21.x)))));
          ((float2*)(&__33))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__34))[0]);
          float2 v__22 = make_float2(broadcast_var_5, broadcast_var_5);
          __32.x = (__33.x*v__22.x);
          __32.y = (__33.y*v__22.y);
        __31.x = exp2f(__32.x);
        __31.y = exp2f(__32.y);
        __30.x = (v__19.x*__31.x);
        __30.y = (v__19.y*__31.y);
      __28.x = (__29.x-__30.x);
      __28.y = (__29.y-__30.y);
    (reinterpret_cast<__nv_bfloat162*>(&__27))[0] = __float22bfloat162_rn(((float2*)(&__28))[0]);
    *(uint1*)(v_new_c_local_cast_10 + 0) = __27;
    *(uint1*)(((bfloat16_t*)v_new_c) + ((((((((int)threadIdx.x) >> 5) * 256) + ((i_20 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(v_new_c_local_cast_10 + 0);
  }
  __syncthreads();
  *(uint4*)(v_new + ((((((((int)threadIdx.x) >> 1) * 2048) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 8257536)) = *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  *(uint4*)(v_new_c_local_cast_13 + 0) = *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    uint2 __35;
    float4 __36;
      float4 __37;
      uint2 v__23 = *(uint2*)(v_new_c_local_cast_13 + (i_21 * 4));
      ((float2*)(&__37))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__23))[0]);
      ((float2*)(&__37))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__23))[1]);
      float4 v__24 = make_float4(exp2f((((float)(((bfloat16_t*)g_c)[127] - ((bfloat16_t*)g_c)[((((int)threadIdx.x) >> 1) + 64)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[127] - ((bfloat16_t*)g_c)[((((int)threadIdx.x) >> 1) + 64)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[127] - ((bfloat16_t*)g_c)[((((int)threadIdx.x) >> 1) + 64)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)(((bfloat16_t*)g_c)[127] - ((bfloat16_t*)g_c)[((((int)threadIdx.x) >> 1) + 64)])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
      __36.x = (__37.x*v__24.x);
      __36.y = (__37.y*v__24.y);
      __36.z = (__37.z*v__24.z);
      __36.w = (__37.w*v__24.w);
    (reinterpret_cast<__nv_bfloat162*>(&__35))[0] = __float22bfloat162_rn(((float2*)(&__36))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__35))[1] = __float22bfloat162_rn(((float2*)(&__36))[1]);
    *(uint2*)(v_new_c_local_cast_13 + (i_21 * 4)) = __35;
  }
  *(uint4*)(((bfloat16_t*)v_new_c) + (((((int)threadIdx.x) >> 1) * 16) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(v_new_c_local_cast_13 + 0);
  #pragma unroll
  for (int i_22 = 0; i_22 < 8; ++i_22) {
    *(uint1*)(h_c_local_cast_14 + 0) = *(uint1*)(((bfloat16_t*)h_c) + (((((((i_22 >> 2) * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_22 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_22 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    float2 __38;
      float2 __39;
      uint1 v__25 = *(uint1*)(h_c_local_cast_14 + 0);
      ((float2*)(&__39))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__25))[0]);
      float2 v__26 = make_float2(exp2f((((float)((bfloat16_t*)g_c)[127]) * 0x1.71547652b82fep+0f/*1.442695e+00*/)), exp2f((((float)((bfloat16_t*)g_c)[127]) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
      __38.x = (__39.x*v__26.x);
      __38.y = (__39.y*v__26.y);
    *(float2*)(h_next_frag + (i_22 * 2)) = __38;
  }
  {
    tl::GmmaDescriptor desc_a_5;
    tl::GmmaDescriptor desc_b_5;
    __syncthreads();
    tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a_5, (&(((bfloat16_t*)k_c)[8192])));
    tl::initialize_wgmma_descriptor<3, 0, 16>(desc_b_5, (&(((bfloat16_t*)v_new_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_next_frag + 0), 16);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 16, 16, true, true, 1, 1>(uint64_t(desc_a_5 + (((i_23 * 8192) + (ki_5 * 2048)) >> 4)), uint64_t(desc_b_5 + ((ki_5 * 512) >> 4)), ((uint32_t*)(h_next_frag + (i_23 * 8))), 1);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_next_frag + 0), 16);
  }
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 2; ++i_24) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_c)[((((i_24 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), __pack_half2(((bfloat16_t)h_next_frag[(i_24 * 8)]), ((bfloat16_t)h_next_frag[((i_24 * 8) + 1)])), __pack_half2(((bfloat16_t)h_next_frag[((i_24 * 8) + 2)]), ((bfloat16_t)h_next_frag[((i_24 * 8) + 3)])), __pack_half2(((bfloat16_t)h_next_frag[((i_24 * 8) + 4)]), ((bfloat16_t)h_next_frag[((i_24 * 8) + 5)])), __pack_half2(((bfloat16_t)h_next_frag[((i_24 * 8) + 6)]), ((bfloat16_t)h_next_frag[((i_24 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    *(uint4*)(S + ((((((((int)blockIdx.z) * 1064960) + (i_25 * 8192)) + ((((int)threadIdx.x) >> 1) * 128)) + (((int)blockIdx.x) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1048576)) = *(uint4*)(((bfloat16_t*)h_c) + (((i_25 * 1024) + ((((int)threadIdx.x) >> 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  }
}
