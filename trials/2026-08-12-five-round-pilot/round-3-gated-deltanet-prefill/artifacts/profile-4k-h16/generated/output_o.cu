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

extern "C" __global__ void output_o_bthd_kernel(const bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ g, const bfloat16_t* __restrict__ k, bfloat16_t* __restrict__ o, const bfloat16_t* __restrict__ q, const bfloat16_t* __restrict__ v_new);
extern "C" __global__ void __launch_bounds__(256, 1) output_o_bthd_kernel(const bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ g, const bfloat16_t* __restrict__ k, bfloat16_t* __restrict__ o, const bfloat16_t* __restrict__ q, const bfloat16_t* __restrict__ v_new) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* attn = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_c = ((void*)((char*)buf_dyn_shmem + 0));
  void* q_c = ((void*)((char*)buf_dyn_shmem + 16384));
  void* g_c = ((void*)((char*)buf_dyn_shmem + 32768));
  void* h_c = ((void*)((char*)buf_dyn_shmem + 33792));
  void* v_new_c = ((void*)((char*)buf_dyn_shmem + 66560));
  float o_frag[32];
  bfloat16_t g_c_local_cast[2];
  float attn_frag[16];
  bfloat16_t o_local_cast_1[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((bfloat16_t*)q_c) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q + (((((((int)blockIdx.x) * 131072) + (i * 32768)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    *(uint4*)(((bfloat16_t*)k_c) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(k + (((((((int)blockIdx.x) * 131072) + (i * 32768)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  if (((int)threadIdx.x) < 64) {
    ((bfloat16_t*)g_c)[((int)threadIdx.x)] = g[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 16)) + ((int)blockIdx.z))];
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(uint4*)(((bfloat16_t*)h_c) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(S + ((((((int)blockIdx.z) * 1064960) + (((int)blockIdx.x) * 16384)) + (i_1 * 2048)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    *(uint4*)(((bfloat16_t*)v_new_c) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(v_new + (((((((int)blockIdx.x) * 131072) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(o_frag + (i_3 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  {
    tl::GmmaDescriptor desc_a;
    tl::GmmaDescriptor desc_b;
    __syncthreads();
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)q_c)[0])));
    tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b, (&(((bfloat16_t*)h_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_frag + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((((int)threadIdx.x) >> 7) * 16384) + (ki * 2048)) >> 4)), ((uint32_t*)(o_frag + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_frag + 0), 32);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    *(uint1*)(g_c_local_cast + 0) = make_uint1(__pack_nv_bfloat162(((bfloat16_t*)g_c)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((bfloat16_t*)g_c)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
    float broadcast_var_1 = 0x1.71547652b82fep+0f/*1.442695e+00*/;
    float2 __1;
      float2 v_ = *(float2*)(o_frag + (i_4 * 2));
      float2 __2;
      float2 __3;
        float2 __4;
        uint1 v__1 = *(uint1*)(g_c_local_cast + 0);
        ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        float2 v__2 = make_float2(broadcast_var_1, broadcast_var_1);
        __3.x = (__4.x*v__2.x);
        __3.y = (__4.y*v__2.y);
      __2.x = exp2f(__3.x);
      __2.y = exp2f(__3.y);
      __1.x = (v_.x*__2.x);
      __1.y = (v_.y*__2.y);
    *(float2*)(o_frag + (i_4 * 2)) = __1;
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(attn_frag + (i_5 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  {
    tl::GmmaDescriptor desc_a_1;
    tl::GmmaDescriptor desc_b_1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)q_c)[0])));
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((bfloat16_t*)k_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(attn_frag + 0), 16);
    tl::warpgroup_arrive();
    #pragma unroll
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + (((((ki_1 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(attn_frag + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(attn_frag + 0), 16);
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    float condval;
    if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) <= (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval = (attn_frag[i_6] * exp2f((((float)(((bfloat16_t*)g_c)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] - ((bfloat16_t*)g_c)[(((((((int)threadIdx.x) >> 7) * 32) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1))])) * 0x1.71547652b82fep+0f/*1.442695e+00*/)));
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    ((bfloat16_t*)attn)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_6 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_6 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_6 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1))] = ((bfloat16_t)condval);
  }
  {
    tl::GmmaDescriptor desc_a_2;
    tl::GmmaDescriptor desc_b_2;
    __syncthreads();
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)attn)[0])));
    tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((bfloat16_t*)v_new_c)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_frag + 0), 32);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_2 * 2048)) >> 4)), ((uint32_t*)(o_frag + 0)), 1);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_frag + 0), 32);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    uint1 __5;
    float2 v__3 = *(float2*)(o_frag + (i_7 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
    *(uint1*)(o_local_cast_1 + 0) = __5;
    *(uint1*)(o + ((((((((((int)blockIdx.x) * 131072) + (((((int)threadIdx.x) & 127) >> 5) * 32768)) + ((i_7 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(o_local_cast_1 + 0);
  }
}
