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

extern "C" __global__ void tilelang_fused_chunk_gdr_fwd_kernel_kernel(__grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cp_seq_map, const int* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, __grid_constant__ const CUtensorMap q_desc, const int* __restrict__ raw_cu_seqlens, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) tilelang_fused_chunk_gdr_fwd_kernel_kernel(__grid_constant__ const CUtensorMap a_desc, const bfloat16_t* __restrict__ b, const int* __restrict__ chunk_offsets, const int* __restrict__ cp_seq_map, const int* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, __grid_constant__ const CUtensorMap q_desc, const int* __restrict__ raw_cu_seqlens, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 131072));
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 147456));
  void* vd_shared = ((void*)((char*)buf_dyn_shmem + 163840));
  void* vn_shared = ((void*)((char*)buf_dyn_shmem + 180224));
  void* p_shared = ((void*)((char*)buf_dyn_shmem + 196608));
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[128];
  __shared__ __align__(16) float g_exp_shared[64];
  __shared__ __align__(16) float g_shared[128];
  __shared__ __align__(16) float b_shared[128];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[64];
  float u_fragment[64];
  bfloat16_t v_shared_local_cast[2];
  bfloat16_t v_shared_local_cast_1[2];
  float v_fragment[64];
  float p_fragment[32];
  float g_fragment[32];
  float a_fragment[32];
  bfloat16_t a_shared_local_cast_2[2];
  bfloat16_t a_shared_local_cast_3[2];
  float o_fragment[64];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = cu_seqlens[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)];
  seq_end_idx = cu_seqlens[((((int64_t)((int)blockIdx.x)) >> (int64_t)6) + (int64_t)1)];
  chunk_start_idx = chunk_offsets[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)];
  raw_batch_idx = cp_seq_map[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)];
  int condval;
  if (((-1 <= raw_batch_idx) && (raw_batch_idx < raw_batch_size))) {
    condval = raw_cu_seqlens[(((int64_t)raw_batch_idx) + (int64_t)1)];
  } else {
    condval = 0;
  }
  raw_seq_end_idx = condval;
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 63) - seq_start_idx) >> 6);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 6);
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + (((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)i) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((((int64_t)i) & (int64_t)31) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
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
      g_last_local[0] = g_exp_shared[63];
      #pragma unroll
      for (int i_2 = 0; i_2 < 128; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local[0]);
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 256, 64>(desc_a, (&(((bfloat16_t*)k_shared)[((i_s & 1) * 8192)])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)vn_shared)[0])));
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
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 64; ++i_4) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + ((((((((((int64_t)raw_batch_idx) * (int64_t)1048576) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)16384)) + ((((int64_t)i_4) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((((int64_t)i_4) & (int64_t)31) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_4 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial(3, 128);
        if (((int)threadIdx.x) < 192) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 64) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval_1;
          if (((((i_s_1 * 64) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval_1 = exp2f(((g_shared[(((i_s_1 & 1) * 64) + 63)] - g_shared[((((i_s_1 & 1) * 64) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval_1;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          tl::GmmaDescriptor desc_a_1;
          tl::GmmaDescriptor desc_b_1;
          tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)k_shared)[((i_s_1 & 1) * 8192)])));
          tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_1, (&(((bfloat16_t*)h_shared)[0])));
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(u_fragment + 0), 64);
          tl::warpgroup_arrive();
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((ki_1 * 2048) >> 4)), ((uint32_t*)(u_fragment + 0)), ((0 < ki_1) ? 1 : 0));
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(u_fragment + 0), 64);
        }
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int i_5 = 0; i_5 < 32; ++i_5) {
          float2 __1;
            float2 v_ = *(float2*)(u_fragment + (i_5 * 2));
            float2 v__1 = make_float2((g_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] * -0x1p+0f/*-1.000000e+00*/));
            __1.x = (v_.x*v__1.x);
            __1.y = (v_.y*v__1.y);
          *(float2*)(u_fragment + (i_5 * 2)) = __1;
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)v_shared) + (((((((((((i_s_1 & 1) * 8192) + ((i_6 >> 4) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_6 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096));
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_6 * 2));
            float2 __3;
            uint1 v__3 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            __2.x = (v__2.x+__3.x);
            __2.y = (v__2.y+__3.y);
          *(float2*)(u_fragment + (i_6 * 2)) = __2;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_7 = 0; i_7 < 32; ++i_7) {
          uint1 __4;
          float2 v__4 = *(float2*)(u_fragment + (i_7 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __4;
          *(uint1*)(((bfloat16_t*)v_shared) + (((((((((((i_s_1 & 1) * 8192) + ((i_7 >> 4) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_7 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 4096)) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          tl::GmmaDescriptor desc_a_2;
          tl::GmmaDescriptor desc_b_2;
          tl::__sync_thread_partial(4, 128);
          tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)a_shared)[((i_s_1 & 1) * 4096)])));
          tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((bfloat16_t*)v_shared)[((i_s_1 & 1) * 8192)])));
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(v_fragment + 0), 64);
          tl::warpgroup_arrive();
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_2 + ((ki_2 * 32) >> 4)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), ((uint32_t*)(v_fragment + 0)), ((0 < ki_2) ? 1 : 0));
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(v_fragment + 0), 64);
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_8 = 0; i_8 < 8; ++i_8) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vd_shared)[(((((i_8 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)v_fragment[(i_8 * 8)]), ((bfloat16_t)v_fragment[((i_8 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_8 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_8 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_8 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_8 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_8 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_8 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_9 = 0; i_9 < 32; ++i_9) {
          float2 __5;
            float2 v__5 = *(float2*)(v_fragment + (i_9 * 2));
            float2 v__6 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)], g_rev_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)]);
            __5.x = (v__5.x*v__6.x);
            __5.y = (v__5.y*v__6.y);
          *(float2*)(v_fragment + (i_9 * 2)) = __5;
        }
        tl::__sync_thread_partial(4, 128);
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vn_shared)[(((((i_10 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)v_fragment[(i_10 * 8)]), ((bfloat16_t)v_fragment[((i_10 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_10 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_10 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_10 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_10 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_10 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_10 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            tl::GmmaDescriptor desc_a_3;
            tl::GmmaDescriptor desc_b_3;
            tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((bfloat16_t*)q_shared)[((i_s_2 & 1) * 8192)])));
            tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((bfloat16_t*)k_shared)[((i_s_2 & 1) * 8192)])));
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(p_fragment + 0), 32);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(p_fragment + 0)), ((0 < ki_3) ? 1 : 0));
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(p_fragment + 0), 32);
          }
          #pragma unroll
          for (int i_11 = 0; i_11 < 16; ++i_11) {
            float2 __6;
              float2 v__7 = make_float2(g_shared[((((((i_s_2 & 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)], g_shared[((((((i_s_2 & 1) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)]);
              float2 v__8 = *(float2*)(g_shared + ((((i_s_2 & 1) * 64) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
              __6.x = (v__7.x-v__8.x);
              __6.y = (v__7.y-v__8.y);
            *(float2*)(g_fragment + (i_11 * 2)) = __6;
          }
          #pragma unroll
          for (int i_12 = 0; i_12 < 32; ++i_12) {
            if ((((((i_12 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_12 & 1)) + 128) <= ((((((int)threadIdx.x) >> 5) * 16) + (((i_12 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) {
              g_fragment[i_12] = exp2f((g_fragment[i_12] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_12] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_13 = 0; i_13 < 16; ++i_13) {
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((i_s_2 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192));
            float2 __7;
            uint1 v__9 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
            *(float2*)(a_fragment + (i_13 * 2)) = __7;
          }
          #pragma unroll
          for (int i_14 = 0; i_14 < 32; ++i_14) {
            a_fragment[i_14] = (a_fragment[i_14] * g_fragment[i_14]);
          }
          #pragma unroll
          for (int i_15 = 0; i_15 < 16; ++i_15) {
            float2 __8;
              float2 v__10 = *(float2*)(a_fragment + (i_15 * 2));
              float2 v__11 = *(float2*)(b_shared + ((((i_s_2 & 1) * 64) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
              __8.x = (v__10.x*v__11.x);
              __8.y = (v__10.y*v__11.y);
            *(float2*)(a_fragment + (i_15 * 2)) = __8;
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_16 = 0; i_16 < 16; ++i_16) {
            uint1 __9;
            float2 v__12 = *(float2*)(a_fragment + (i_16 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __9;
            *(uint1*)(((bfloat16_t*)a_shared) + ((((((((((i_s_2 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_16 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_16 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_16 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_16 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            tl::GmmaDescriptor desc_a_4;
            tl::GmmaDescriptor desc_b_4;
            tl::__sync_thread_partial(5, 128);
            tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((bfloat16_t*)q_shared)[((i_s_2 & 1) * 8192)])));
            tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_4, (&(((bfloat16_t*)h_shared)[0])));
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_fragment + 0), 64);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_4 + ((((ki_4 >> 2) * 8192) + ((ki_4 & 3) * 32)) >> 4)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), ((uint32_t*)(o_fragment + 0)), ((0 < ki_4) ? 1 : 0));
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_fragment + 0), 64);
          }
          #pragma unroll
          for (int i_17 = 0; i_17 < 32; ++i_17) {
            p_fragment[i_17] = (p_fragment[i_17] * g_fragment[i_17]);
          }
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)p_shared)[(((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_18 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)p_fragment[(i_18 * 8)]), ((bfloat16_t)p_fragment[((i_18 * 8) + 1)])), __pack_half2(((bfloat16_t)p_fragment[((i_18 * 8) + 2)]), ((bfloat16_t)p_fragment[((i_18 * 8) + 3)])), __pack_half2(((bfloat16_t)p_fragment[((i_18 * 8) + 4)]), ((bfloat16_t)p_fragment[((i_18 * 8) + 5)])), __pack_half2(((bfloat16_t)p_fragment[((i_18 * 8) + 6)]), ((bfloat16_t)p_fragment[((i_18 * 8) + 7)])));
          }
          bar_3[0].arrive();
          #pragma unroll
          for (int i_19 = 0; i_19 < 32; ++i_19) {
            float2 __10;
              float2 v__13 = *(float2*)(o_fragment + (i_19 * 2));
              float2 v__14 = make_float2(g_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)], g_exp_shared[(((((((int)threadIdx.x) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)]);
              __10.x = (v__13.x*v__14.x);
              __10.y = (v__13.y*v__14.y);
            *(float2*)(o_fragment + (i_19 * 2)) = __10;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            tl::GmmaDescriptor desc_a_5;
            tl::GmmaDescriptor desc_b_5;
            tl::__sync_thread_partial(5, 128);
            tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((bfloat16_t*)p_shared)[0])));
            tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((bfloat16_t*)vd_shared)[0])));
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_fragment + 0), 64);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
            #pragma unroll
            for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
              tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 32) >> 4)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), ((uint32_t*)(o_fragment + 0)), 1);
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(o_fragment + 0), 64);
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial(5, 128);
          #pragma unroll
          for (int i_20 = 0; i_20 < 8; ++i_20) {
            tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)o_shared)[(((((i_20 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)o_fragment[(i_20 * 8)]), ((bfloat16_t)o_fragment[((i_20 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_20 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_20 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_20 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_20 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_20 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_20 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 64) + seq_start_idx);
            if (tl::tl_shuffle_elect<32>()) {
              data_is_ready[(i_s_3 & 1)].expect_transaction(16384);
              tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)q_shared)[((i_s_3 & 1) * 8192)])), 0, left, (((int)blockIdx.x) & 63), batch_idx);
              tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)q_shared)[(((i_s_3 & 1) * 8192) + 4096)])), 64, left, (((int)blockIdx.x) & 63), batch_idx);
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
                tl::fence_proxy_async();
                tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)v_shared)[((i_s_4 & 1) * 8192)])), 0, left_1, (((int)blockIdx.x) & 63), batch_idx);
                tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)v_shared)[(((i_s_4 & 1) * 8192) + 4096)])), 64, left_1, (((int)blockIdx.x) & 63), batch_idx);
              }
              if ((left_1 + 64) <= seq_end_idx) {
                #pragma unroll
                for (int i_21 = 0; i_21 < 2; ++i_21) {
                  bfloat16_t condval_2;
                  if (((((13 <= (((left_1 + ((int)threadIdx.x)) >> 5) + i_21)) && ((((i_21 * 32) + left_1) + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_2 = b[((((((((int64_t)i_21) * (int64_t)2048) + (((int64_t)left_1) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)26624)];
                  } else {
                    condval_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  }
                  b_shared[(((((i_s_4 & 1) * 64) + (i_21 * 32)) + ((int)threadIdx.x)) - 416)] = ((float)condval_2);
                }
              } else {
                #pragma unroll
                for (int i_22 = 0; i_22 < 2; ++i_22) {
                  if ((((i_22 * 32) + left_1) + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                    bfloat16_t condval_3;
                    if (((((13 <= (((left_1 + ((int)threadIdx.x)) >> 5) + i_22)) && ((((i_22 * 32) + left_1) + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_3 = b[((((((((int64_t)i_22) * (int64_t)2048) + (((int64_t)left_1) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)26624)];
                    } else {
                      condval_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    }
                    b_shared[(((((i_s_4 & 1) * 64) + (i_22 * 32)) + ((int)threadIdx.x)) - 416)] = ((float)condval_3);
                  } else {
                    b_shared[(((((i_s_4 & 1) * 64) + (i_22 * 32)) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                  }
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 64) + seq_start_idx);
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_5 & 1)].expect_transaction(8192);
                  tl::fence_proxy_async();
                  tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)a_shared)[((i_s_5 & 1) * 4096)])), 0, left_2, (((int)blockIdx.x) & 63), batch_idx);
                }
                if ((left_2 + 64) <= seq_end_idx) {
                  #pragma unroll
                  for (int i_23 = 0; i_23 < 2; ++i_23) {
                    bfloat16_t condval_4;
                    if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_23)) && ((((i_23 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_4 = g[((((((((int64_t)i_23) * (int64_t)2048) + (((int64_t)left_2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)28672)];
                    } else {
                      condval_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    }
                    g_shared[(((((i_s_5 & 1) * 64) + (i_23 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_4);
                  }
                } else {
                  #pragma unroll
                  for (int i_24 = 0; i_24 < 2; ++i_24) {
                    if ((((i_24 * 32) + left_2) + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                      bfloat16_t condval_5;
                      if (((((14 <= (((left_2 + ((int)threadIdx.x)) >> 5) + i_24)) && ((((i_24 * 32) + left_2) + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_5 = g[((((((((int64_t)i_24) * (int64_t)2048) + (((int64_t)left_2) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)28672)];
                      } else {
                        condval_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      }
                      g_shared[(((((i_s_5 & 1) * 64) + (i_24 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_5);
                    } else {
                      bfloat16_t condval_6;
                      if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_6 = g[((((((int64_t)seq_end_idx) * (int64_t)64) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)64)) + (((int64_t)((int)blockIdx.x)) & (int64_t)63)) - (int64_t)64)];
                      } else {
                        condval_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      }
                      g_shared[(((((i_s_5 & 1) * 64) + (i_24 * 32)) + ((int)threadIdx.x)) - 448)] = ((float)condval_6);
                    }
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 64) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[0])), 0, (right - 64), (((int)blockIdx.x) & 63), batch_idx);
                    tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[4096])), 64, (right - 64), (((int)blockIdx.x) & 63), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 64) + seq_start_idx);
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[0])), 0, (seq_split_idx - 64), (((int)blockIdx.x) & 63), batch_idx);
                    tl::tma_store(o_desc, (&(((bfloat16_t*)o_shared)[4096])), 64, (seq_split_idx - 64), (((int)blockIdx.x) & 63), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0, true>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
              }
              seq_split_idx = (((num_iters * 64) + seq_start_idx) - 64);
              bar_o[0].wait(0);
              if (0 <= batch_idx) {
                #pragma unroll
                for (int i_25 = 0; i_25 < 32; ++i_25) {
                  if ((((i_25 * 2) + (((int)threadIdx.x) >> 4)) + seq_split_idx) < (seq_end_idx + 30)) {
                    if (15 <= ((((((int)threadIdx.x) >> 4) + seq_split_idx) >> 1) + i_25)) {
                      if ((((i_25 * 2) + (((int)threadIdx.x) >> 4)) + seq_split_idx) < (num_tokens + 30)) {
                        if (batch_idx < 1) {
                          *(uint4*)(o + (((((((((int64_t)i_25) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)seq_split_idx) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)245760)) = *(uint4*)(((bfloat16_t*)o_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((i_25 >> 2) * 512)) + (((((i_25 * 2) + (((int)threadIdx.x) >> 4)) + 2) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_25 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
