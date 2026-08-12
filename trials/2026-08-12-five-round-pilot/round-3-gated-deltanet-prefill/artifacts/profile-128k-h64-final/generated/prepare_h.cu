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

extern "C" __global__ void tilelang_prepare_h_kernel_kernel(__grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ g, bfloat16_t* __restrict__ ht, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ mt, const int* __restrict__ num_warmup_chunks, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_prepare_h_kernel_kernel(__grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ g, bfloat16_t* __restrict__ ht, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ mt, const int* __restrict__ num_warmup_chunks, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* m_shared_L = ((void*)((char*)buf_dyn_shmem + 114688));
  void* m_shared_R = ((void*)((char*)buf_dyn_shmem + 131072));
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 147456));
  void* y_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  void* z_shared_L = ((void*)((char*)buf_dyn_shmem + 180224));
  void* z_shared_R = ((void*)((char*)buf_dyn_shmem + 188416));
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t bar_2_mem[1];
  auto bar_2 = reinterpret_cast<Barrier*>(bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int num_iters = 0;
  signed char calc_mt = (signed char)0;
  float h_fragment[128];
  __shared__ __align__(16) float g_shared[128];
  float m_fragment_R[64];
  float g_prod_X[1];
  __shared__ __align__(16) float b_shared[128];
  float g_last_local_X[1];
  float m_fragment_L[64];
  float g_prod_Y[1];
  float g_last_local_Y[1];
  float g_last_local_S[1];
  bfloat16_t ht_local_cast[2];
  float x_fragment[64];
  float z_fragment_R[32];
  bfloat16_t mt_local_cast_1[2];
  __shared__ __align__(16) float g_rev_exp_shared[64];
  float y_fragment[64];
  bfloat16_t v_shared_local_cast_2[2];
  float g_rev_exp_shared_local_cast_3[2];
  float z_fragment_L[32];
  bfloat16_t mt_local_cast_4[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_0[0].init(416);
    bar_1[0].init(256);
    bar_2[0].init(384);
    bar_3[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = cu_seqlens[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)];
  seq_end_idx = cu_seqlens[((((int64_t)((int)blockIdx.x)) >> (int64_t)6) + (int64_t)1)];
  chunk_start_idx = chunk_offsets[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)];
  num_iters = num_warmup_chunks[((int64_t)((int)blockIdx.x))];
  calc_mt = ((signed char)((((seq_end_idx + 63) - seq_start_idx) >> 6) <= num_iters));
  seq_start_idx = (seq_end_idx - (num_iters * 64));
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<168>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(h_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)h_shared)[(((((((i_1 & 7) >> 2) * 8192) + ((i_1 >> 3) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local_S[0] = exp2f((g_shared[(((i_s & 1) * 64) + 63)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
      #pragma unroll
      for (int i_2 = 0; i_2 < 128; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local_S[0]);
      }
      bar_2[0].arrive();
      bar_2[0].wait((i_s & 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a, (&(((bfloat16_t*)x_shared)[0])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)y_shared)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_fragment + 0), 128);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, true, true, 1, 1>(uint64_t(desc_a + (((i_3 * 8192) + (ki * 2048)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(h_fragment + (i_3 * 64))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_fragment + 0), 128);
      }
      bar_3[0].arrive();
      data_is_free[(i_s & 1)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      uint1 __1;
      float2 v_ = *(float2*)(h_fragment + (i_4 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      *(uint1*)(ht_local_cast + 0) = __1;
      *(uint1*)(ht + (((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)i_4) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((((int64_t)i_4) & (int64_t)31) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(ht_local_cast + 0);
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<160>();
      if ((bool)calc_mt) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 64; ++i_5) {
          if (((((((i_5 >> 5) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64) == ((((((i_5 & 31) >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) + 64)) {
            m_fragment_R[i_5] = 0x1p+0f/*1.000000e+00*/;
          } else {
            m_fragment_R[i_5] = 0x0p+0f/*0.000000e+00*/;
          }
        }
        g_prod_X[0] = 0x0p+0f/*0.000000e+00*/;
      }
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        {
          tl::GmmaDescriptor desc_a_1;
          tl::GmmaDescriptor desc_b_1;
          tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_1, (&(((bfloat16_t*)a_shared)[((i_s_1 & 1) * 4096)])));
          tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((bfloat16_t*)k_shared)[((i_s_1 & 1) * 8192)])));
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(x_fragment + 0), 64);
          tl::warpgroup_arrive();
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, true, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 2048) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(x_fragment + 0)), ((0 < ki_1) ? 1 : 0));
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(x_fragment + 0), 64);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          float2 __2;
            float2 v__1 = *(float2*)(x_fragment + (i_6 * 2));
            float2 v__2 = make_float2((b_shared[((((((i_s_1 & 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_6 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] * -0x1p+0f/*-1.000000e+00*/), (b_shared[((((((i_s_1 & 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_6 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] * -0x1p+0f/*-1.000000e+00*/));
            __2.x = (v__1.x*v__2.x);
            __2.y = (v__1.y*v__2.y);
          *(float2*)(x_fragment + (i_6 * 2)) = __2;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)x_shared)[(((((i_7 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)x_fragment[(i_7 * 8)]), ((bfloat16_t)x_fragment[((i_7 * 8) + 1)])), __pack_half2(((bfloat16_t)x_fragment[((i_7 * 8) + 2)]), ((bfloat16_t)x_fragment[((i_7 * 8) + 3)])), __pack_half2(((bfloat16_t)x_fragment[((i_7 * 8) + 4)]), ((bfloat16_t)x_fragment[((i_7 * 8) + 5)])), __pack_half2(((bfloat16_t)x_fragment[((i_7 * 8) + 6)]), ((bfloat16_t)x_fragment[((i_7 * 8) + 7)])));
        }
        bar_2[0].arrive();
        if ((bool)calc_mt) {
          g_prod_X[0] = (g_prod_X[0] + g_shared[(((i_s_1 & 1) * 64) + 63)]);
          tl::__sync_thread_partial(4, 128);
          #pragma unroll
          for (int i_8 = 0; i_8 < 8; ++i_8) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)m_shared_R)[((((((i_8 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) - 4096)])), __pack_half2(((bfloat16_t)m_fragment_R[(i_8 * 8)]), ((bfloat16_t)m_fragment_R[((i_8 * 8) + 1)])), __pack_half2(((bfloat16_t)m_fragment_R[((i_8 * 8) + 2)]), ((bfloat16_t)m_fragment_R[((i_8 * 8) + 3)])), __pack_half2(((bfloat16_t)m_fragment_R[((i_8 * 8) + 4)]), ((bfloat16_t)m_fragment_R[((i_8 * 8) + 5)])), __pack_half2(((bfloat16_t)m_fragment_R[((i_8 * 8) + 6)]), ((bfloat16_t)m_fragment_R[((i_8 * 8) + 7)])));
          }
          bar_3[0].wait((i_s_1 & 1));
          {
            tl::GmmaDescriptor desc_a_2;
            tl::GmmaDescriptor desc_b_2;
            tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)k_shared)[((i_s_1 & 1) * 8192)])));
            tl::__sync_thread_partial(4, 128);
            tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_2, (&(((bfloat16_t*)m_shared_R)[0])));
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(z_fragment_R + 0), 32);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(z_fragment_R + 0)), ((0 < ki_2) ? 1 : 0));
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(z_fragment_R + 0), 32);
          }
          tl::__sync_thread_partial(4, 128);
          #pragma unroll
          for (int i_9 = 0; i_9 < 4; ++i_9) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)z_shared_R)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)z_fragment_R[(i_9 * 8)]), ((bfloat16_t)z_fragment_R[((i_9 * 8) + 1)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_9 * 8) + 2)]), ((bfloat16_t)z_fragment_R[((i_9 * 8) + 3)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_9 * 8) + 4)]), ((bfloat16_t)z_fragment_R[((i_9 * 8) + 5)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_9 * 8) + 6)]), ((bfloat16_t)z_fragment_R[((i_9 * 8) + 7)])));
          }
          {
            tl::GmmaDescriptor desc_a_3;
            tl::GmmaDescriptor desc_b_3;
            tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a_3, (&(((bfloat16_t*)x_shared)[0])));
            tl::__sync_thread_partial(4, 128);
            tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_3, (&(((bfloat16_t*)z_shared_R)[0])));
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(m_fragment_R + 0), 64);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
            #pragma unroll
            for (int i_10 = 0; i_10 < 2; ++i_10) {
              #pragma unroll
              for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_3 + (((i_10 * 8192) + (ki_3 * 2048)) >> 4)), uint64_t(desc_b_3 + ((ki_3 * 2048) >> 4)), ((uint32_t*)(m_fragment_R + (i_10 * 32))), 1);
              }
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(m_fragment_R + 0), 64);
          }
        }
        data_is_free[(i_s_1 & 1)].arrive();
      }
      if ((bool)calc_mt) {
        g_last_local_X[0] = exp2f((g_prod_X[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
        #pragma unroll
        for (int i_11 = 0; i_11 < 64; ++i_11) {
          m_fragment_R[i_11] = (m_fragment_R[i_11] * g_last_local_X[0]);
        }
        #pragma unroll
        for (int i_12 = 0; i_12 < 32; ++i_12) {
          uint1 __3;
          float2 v__3 = *(float2*)(m_fragment_R + (i_12 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
          *(uint1*)(mt_local_cast_1 + 0) = __3;
          *(uint1*)(mt + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)i_12) >> (int64_t)4) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_12) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((((int64_t)i_12) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)8128)) = *(uint1*)(mt_local_cast_1 + 0);
        }
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<160>();
        if ((bool)calc_mt) {
          #pragma unroll
          for (int i_13 = 0; i_13 < 64; ++i_13) {
            if (((((((i_13 >> 5) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_13 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128) == (((((i_13 & 31) >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_13 & 1))) {
              m_fragment_L[i_13] = 0x1p+0f/*1.000000e+00*/;
            } else {
              m_fragment_L[i_13] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          g_prod_Y[0] = 0x0p+0f/*0.000000e+00*/;
        }
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          g_last_local_Y[0] = g_shared[(((i_s_2 & 1) * 64) + 63)];
          tl::__sync_thread_partial(3, 128);
          if (((int)threadIdx.x) < 320) {
            g_rev_exp_shared[(((int)threadIdx.x) - 256)] = exp2f(((g_last_local_Y[0] - g_shared[((((i_s_2 & 1) * 64) + ((int)threadIdx.x)) - 256)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          }
          g_last_local_Y[0] = exp2f((g_last_local_Y[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          bar_1[0].arrive();
          bar_1[0].wait((i_s_2 & 1));
          {
            tl::GmmaDescriptor desc_a_4;
            tl::GmmaDescriptor desc_b_4;
            tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((bfloat16_t*)k_shared)[((i_s_2 & 1) * 8192)])));
            tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_4, (&(((bfloat16_t*)h_shared)[0])));
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(y_fragment + 0), 64);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), ((uint32_t*)(y_fragment + 0)), ((0 < ki_4) ? 1 : 0));
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(y_fragment + 0), 64);
          }
          #pragma unroll
          for (int i_14 = 0; i_14 < 64; ++i_14) {
            y_fragment[i_14] = (y_fragment[i_14] * g_last_local_Y[0]);
          }
          tl::__sync_thread_partial(3, 128);
          #pragma unroll
          for (int i_15 = 0; i_15 < 32; ++i_15) {
            *(uint1*)(v_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)v_shared) + ((((((((i_s_2 & 1) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_15 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384));
            *(float2*)(g_rev_exp_shared_local_cast_3 + 0) = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)], g_rev_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)]);
            float2 __4;
              float2 v__4 = *(float2*)(y_fragment + (i_15 * 2));
              float2 __5;
                float2 __6;
                uint1 v__5 = *(uint1*)(v_shared_local_cast_2 + 0);
                ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
                float2 v__6 = *(float2*)(g_rev_exp_shared_local_cast_3 + 0);
                __5.x = (__6.x*v__6.x);
                __5.y = (__6.y*v__6.y);
              __4.x = (v__4.x-__5.x);
              __4.y = (v__4.y-__5.y);
            *(float2*)(y_fragment + (i_15 * 2)) = __4;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_16 = 0; i_16 < 8; ++i_16) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)y_shared)[(((((i_16 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)y_fragment[(i_16 * 8)]), ((bfloat16_t)y_fragment[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)y_fragment[((i_16 * 8) + 2)]), ((bfloat16_t)y_fragment[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)y_fragment[((i_16 * 8) + 4)]), ((bfloat16_t)y_fragment[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)y_fragment[((i_16 * 8) + 6)]), ((bfloat16_t)y_fragment[((i_16 * 8) + 7)])));
          }
          bar_2[0].arrive();
          if ((bool)calc_mt) {
            g_prod_Y[0] = (g_prod_Y[0] + g_shared[(((i_s_2 & 1) * 64) + 63)]);
            tl::__sync_thread_partial(5, 128);
            #pragma unroll
            for (int i_17 = 0; i_17 < 8; ++i_17) {
              tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)m_shared_L)[(((((((((i_17 >> 2) * 4) + (((int)threadIdx.x) >> 5)) & 7) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)m_fragment_L[(i_17 * 8)]), ((bfloat16_t)m_fragment_L[((i_17 * 8) + 1)])), __pack_half2(((bfloat16_t)m_fragment_L[((i_17 * 8) + 2)]), ((bfloat16_t)m_fragment_L[((i_17 * 8) + 3)])), __pack_half2(((bfloat16_t)m_fragment_L[((i_17 * 8) + 4)]), ((bfloat16_t)m_fragment_L[((i_17 * 8) + 5)])), __pack_half2(((bfloat16_t)m_fragment_L[((i_17 * 8) + 6)]), ((bfloat16_t)m_fragment_L[((i_17 * 8) + 7)])));
            }
            bar_3[0].wait((i_s_2 & 1));
            {
              tl::GmmaDescriptor desc_a_5;
              tl::GmmaDescriptor desc_b_5;
              tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((bfloat16_t*)k_shared)[((i_s_2 & 1) * 8192)])));
              tl::__sync_thread_partial(5, 128);
              tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_5, (&(((bfloat16_t*)m_shared_L)[0])));
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(z_fragment_L + 0), 32);
              tl::warpgroup_arrive();
              tl::fence_proxy_async();
              #pragma unroll
              for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_5 + ((((ki_5 >> 2) * 8192) + ((ki_5 & 3) * 32)) >> 4)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), ((uint32_t*)(z_fragment_L + 0)), ((0 < ki_5) ? 1 : 0));
              }
              tl::warpgroup_commit_batch();
              tl::warpgroup_wait<0>();
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(z_fragment_L + 0), 32);
            }
            tl::__sync_thread_partial(5, 128);
            #pragma unroll
            for (int i_18 = 0; i_18 < 4; ++i_18) {
              tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)z_shared_L)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_18 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)z_fragment_L[(i_18 * 8)]), ((bfloat16_t)z_fragment_L[((i_18 * 8) + 1)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_18 * 8) + 2)]), ((bfloat16_t)z_fragment_L[((i_18 * 8) + 3)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_18 * 8) + 4)]), ((bfloat16_t)z_fragment_L[((i_18 * 8) + 5)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_18 * 8) + 6)]), ((bfloat16_t)z_fragment_L[((i_18 * 8) + 7)])));
            }
            {
              tl::GmmaDescriptor desc_a_6;
              tl::GmmaDescriptor desc_b_6;
              tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a_6, (&(((bfloat16_t*)x_shared)[0])));
              tl::__sync_thread_partial(5, 128);
              tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b_6, (&(((bfloat16_t*)z_shared_L)[0])));
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(m_fragment_L + 0), 64);
              tl::warpgroup_arrive();
              tl::fence_proxy_async();
              #pragma unroll
              for (int i_19 = 0; i_19 < 2; ++i_19) {
                #pragma unroll
                for (int ki_6 = 0; ki_6 < 4; ++ki_6) {
                  tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_6 + (((i_19 * 8192) + (ki_6 * 2048)) >> 4)), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), ((uint32_t*)(m_fragment_L + (i_19 * 32))), 1);
                }
              }
              tl::warpgroup_commit_batch();
              tl::warpgroup_wait<0>();
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(m_fragment_L + 0), 64);
            }
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        if ((bool)calc_mt) {
          g_last_local_Y[0] = exp2f((g_prod_Y[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          #pragma unroll
          for (int i_20 = 0; i_20 < 64; ++i_20) {
            m_fragment_L[i_20] = (m_fragment_L[i_20] * g_last_local_Y[0]);
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 32; ++i_21) {
            uint1 __7;
            float2 v__7 = *(float2*)(m_fragment_L + (i_21 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__7))[0]);
            *(uint1*)(mt_local_cast_4 + 0) = __7;
            *(uint1*)(mt + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)i_21) >> (int64_t)4) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_21) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((((int64_t)i_21) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)16384)) = *(uint1*)(mt_local_cast_4 + 0);
          }
        }
      } else {
        tl::warpgroup_reg_dealloc<24>();
        if (((int)threadIdx.x) < 416) {
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 64) + seq_start_idx);
            if (tl::tl_shuffle_elect<32>()) {
              data_is_ready[(i_s_3 & 1)].expect_transaction(16384);
              tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)k_shared)[((i_s_3 & 1) * 8192)])), 0, left, (((int)blockIdx.x) & 63), batch_idx);
              tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)k_shared)[(((i_s_3 & 1) * 8192) + 4096)])), 64, left, (((int)blockIdx.x) & 63), batch_idx);
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 64) + seq_start_idx);
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_4 & 1)].expect_transaction(16384);
                tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)v_shared)[((i_s_4 & 1) * 8192)])), 0, left_1, (((int)blockIdx.x) & 63), batch_idx);
                data_is_ready[(i_s_4 & 1)].expect_transaction(8192);
                tl::tma_load(a_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)a_shared)[((i_s_4 & 1) * 4096)])), 0, left_1, (((int)blockIdx.x) & 63), batch_idx);
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 64) + seq_start_idx);
                if ((left_2 + 64) <= seq_end_idx) {
                  #pragma unroll
                  for (int i_22 = 0; i_22 < 2; ++i_22) {
                    bfloat16_t condval;
                    if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_22)) && ((((i_22 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval = g[((((((((int64_t)i_22) * (int64_t)2048) + (((int64_t)left_2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)28672)];
                    } else {
                      condval = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    }
                    g_shared[(((((i_s_5 & 1) * 64) + (i_22 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval);
                  }
                } else {
                  #pragma unroll
                  for (int i_23 = 0; i_23 < 2; ++i_23) {
                    if ((((i_23 * 32) + left_2) + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                      bfloat16_t condval_1;
                      if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_23)) && ((((i_23 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_1 = g[((((((((int64_t)i_23) * (int64_t)2048) + (((int64_t)left_2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)28672)];
                      } else {
                        condval_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      }
                      g_shared[(((((i_s_5 & 1) * 64) + (i_23 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_1);
                    } else {
                      bfloat16_t condval_2;
                      if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_2 = g[((((((int64_t)seq_end_idx) * (int64_t)64) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)64)];
                      } else {
                        condval_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      }
                      g_shared[(((((i_s_5 & 1) * 64) + (i_23 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_2);
                    }
                  }
                }
                if ((left_2 + 64) <= seq_end_idx) {
                  #pragma unroll
                  for (int i_24 = 0; i_24 < 2; ++i_24) {
                    bfloat16_t condval_3;
                    if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_24)) && ((((i_24 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_3 = b[((((((((int64_t)i_24) * (int64_t)2048) + (((int64_t)left_2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)28672)];
                    } else {
                      condval_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    }
                    b_shared[(((((i_s_5 & 1) * 64) + (i_24 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_3);
                  }
                } else {
                  #pragma unroll
                  for (int i_25 = 0; i_25 < 2; ++i_25) {
                    if ((((i_25 * 32) + left_2) + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                      bfloat16_t condval_4;
                      if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_25)) && ((((i_25 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_4 = b[((((((((int64_t)i_25) * (int64_t)2048) + (((int64_t)left_2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)28672)];
                      } else {
                        condval_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      }
                      b_shared[(((((i_s_5 & 1) * 64) + (i_25 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_4);
                    } else {
                      b_shared[(((((i_s_5 & 1) * 64) + (i_25 * 32)) + ((int)threadIdx.x)) - 448)] = 0x0p+0f/*0.000000e+00*/;
                    }
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_iters; ++i_s_6) {
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                bar_1[0].wait((i_s_6 & 1));
              }
            }
          }
        }
      }
    }
  }
}
