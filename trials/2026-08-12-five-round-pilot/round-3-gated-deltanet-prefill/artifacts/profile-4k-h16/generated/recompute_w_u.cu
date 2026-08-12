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

extern "C" __global__ void recompute_w_u_from_A_bthd_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ beta, const bfloat16_t* __restrict__ k, bfloat16_t* __restrict__ u, const bfloat16_t* __restrict__ v, bfloat16_t* __restrict__ w);
extern "C" __global__ void __launch_bounds__(128, 1) recompute_w_u_from_A_bthd_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ beta, const bfloat16_t* __restrict__ k, bfloat16_t* __restrict__ u, const bfloat16_t* __restrict__ v, bfloat16_t* __restrict__ w) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* beta_s = ((void*)((char*)buf_dyn_shmem + 8192));
  void* out_s = ((void*)((char*)buf_dyn_shmem + 8320));
  void* x_s = ((void*)((char*)buf_dyn_shmem + 17408));
  float out_frag[32];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)A_s)[(((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.z) * 65536) + (i * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((bfloat16_t*)beta_s)[((int)threadIdx.x)] = beta[(((((int)blockIdx.z) * 1024) + (((int)threadIdx.x) * 16)) + ((int)blockIdx.y))];
  }
  __syncthreads();
  for (int vt = 0; vt < 2; ++vt) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)x_s)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(v[((((((((int)blockIdx.z) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (vt * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      uint4 __1;
        uint4 v_ = *(uint4*)(((bfloat16_t*)x_s) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        uint4 v__1 = make_uint4(__pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_2 * 16) + (((int)threadIdx.x) >> 3))]));
        *(uint1*)(&(__1.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.x)))));
        *(uint1*)(&(__1.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.y)))));
        *(uint1*)(&(__1.z)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.z))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.z)))));
        *(uint1*)(&(__1.w)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.w))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.w)))));
      *(uint4*)(((bfloat16_t*)x_s) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = __1;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(out_frag + (i_3 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)A_s)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b, (&(((bfloat16_t*)x_s)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(out_frag + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a + ((ki * 32) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(out_frag + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(out_frag + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)out_s)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)out_frag[(i_4 * 8)]), ((bfloat16_t)out_frag[((i_4 * 8) + 1)])), __pack_half2(((bfloat16_t)out_frag[((i_4 * 8) + 2)]), ((bfloat16_t)out_frag[((i_4 * 8) + 3)])), __pack_half2(((bfloat16_t)out_frag[((i_4 * 8) + 4)]), ((bfloat16_t)out_frag[((i_4 * 8) + 5)])), __pack_half2(((bfloat16_t)out_frag[((i_4 * 8) + 6)]), ((bfloat16_t)out_frag[((i_4 * 8) + 7)])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(uint4*)(u + ((((((((int)blockIdx.z) * 131072) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (vt * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((bfloat16_t*)out_s) + (((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
  for (int kt = 0; kt < 2; ++kt) {
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)x_s)[(((((i_6 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(k[((((((((int)blockIdx.z) * 131072) + (i_6 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (kt * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      uint4 __2;
        uint4 v__2 = *(uint4*)(((bfloat16_t*)x_s) + (((((i_7 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        uint4 v__3 = make_uint4(__pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]), __pack_nv_bfloat162(((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))], ((bfloat16_t*)beta_s)[((i_7 * 16) + (((int)threadIdx.x) >> 3))]));
        *(uint1*)(&(__2.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__3.x)))));
        *(uint1*)(&(__2.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__3.y)))));
        *(uint1*)(&(__2.z)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.z))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__3.z)))));
        *(uint1*)(&(__2.w)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.w))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__3.w)))));
      *(uint4*)(((bfloat16_t*)x_s) + (((((i_7 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = __2;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(out_frag + (i_8 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)A_s)[0])));
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_1, (&(((bfloat16_t*)x_s)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(out_frag + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(out_frag + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(out_frag + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)out_s)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)out_frag[(i_9 * 8)]), ((bfloat16_t)out_frag[((i_9 * 8) + 1)])), __pack_half2(((bfloat16_t)out_frag[((i_9 * 8) + 2)]), ((bfloat16_t)out_frag[((i_9 * 8) + 3)])), __pack_half2(((bfloat16_t)out_frag[((i_9 * 8) + 4)]), ((bfloat16_t)out_frag[((i_9 * 8) + 5)])), __pack_half2(((bfloat16_t)out_frag[((i_9 * 8) + 6)]), ((bfloat16_t)out_frag[((i_9 * 8) + 7)])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      *(uint4*)(w + ((((((((int)blockIdx.z) * 131072) + (i_10 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + (kt * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((bfloat16_t*)out_s) + (((((i_10 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}
