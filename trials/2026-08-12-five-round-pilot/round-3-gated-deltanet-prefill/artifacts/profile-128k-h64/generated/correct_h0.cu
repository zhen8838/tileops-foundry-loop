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

extern "C" __global__ void tilelang_correct_h0_kernel_kernel(float* __restrict__ cp_h0, const signed char* __restrict__ fallback_mask, __grid_constant__ const CUtensorMap ht_buffer_desc, const bfloat16_t* __restrict__ mt_buffer, const int* __restrict__ seq_map_r2c, int cp_batch_size, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(256, 1) tilelang_correct_h0_kernel_kernel(float* __restrict__ cp_h0, const signed char* __restrict__ fallback_mask, __grid_constant__ const CUtensorMap ht_buffer_desc, const bfloat16_t* __restrict__ mt_buffer, const int* __restrict__ seq_map_r2c, int cp_batch_size, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* h_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* m_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* hd_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float h_fragment[32];
  bfloat16_t h_shared_local_cast[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(ht_buffer_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  int condval;
  if (((((int)blockIdx.x) >> 8) <= raw_batch_size)) {
    condval = seq_map_r2c[(((int64_t)((int)blockIdx.x)) >> (int64_t)8)];
  } else {
    condval = 0;
  }
  int seq_start_idx = condval;
  if (128 <= ((int)threadIdx.x)) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((i >> 2) == 1) {
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(h_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      }
    }
    if (0 <= seq_start_idx) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        if ((i_1 >> 3) == 1) {
          if (seq_start_idx < cp_batch_size) {
            *(float2*)(cp_h0 + ((((((((((((int64_t)seq_start_idx) * (int64_t)1048576) + (((((int64_t)((int)blockIdx.x)) & (int64_t)255) >> (int64_t)2) * (int64_t)16384)) + ((((int64_t)i_1) >> (int64_t)3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_1) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + (((((int64_t)i_1) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)8192)) = *(float2*)(h_fragment + (i_1 * 2));
          }
        }
      }
    }
  }
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    int condval_1;
    if (((((int)blockIdx.x) >> 8) < raw_batch_size)) {
      condval_1 = seq_map_r2c[((((int64_t)((int)blockIdx.x)) >> (int64_t)8) + (int64_t)1)];
    } else {
      condval_1 = 0;
    }
    int seq_end_idx = condval_1;
    for (int i_s = 0; i_s < ((seq_end_idx - seq_start_idx) - 1); ++i_s) {
      mbarrier[((i_s & 1) + 2)].wait((((i_s & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(i_s & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(ht_buffer_desc, mbarrier[(i_s & 1)], (&(((bfloat16_t*)h_shared)[((i_s & 1) * 4096)])), ((((int)blockIdx.x) & 3) * 32), 0, ((((int)blockIdx.x) & 255) >> 2), (seq_start_idx + i_s));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    int condval_2;
    if (((((int)blockIdx.x) >> 8) < raw_batch_size)) {
      condval_2 = seq_map_r2c[((((int64_t)((int)blockIdx.x)) >> (int64_t)8) + (int64_t)1)];
    } else {
      condval_2 = 0;
    }
    int seq_end_idx_1 = condval_2;
    for (int i_s_1 = 0; i_s_1 < ((seq_end_idx_1 - seq_start_idx) - 1); ++i_s_1) {
      bool condval_3;
      if (((0 <= (seq_start_idx + i_s_1)) && ((seq_start_idx + i_s_1) < cp_batch_size))) {
        condval_3 = ((bool)fallback_mask[(((((int64_t)seq_start_idx) * (int64_t)64) + (((int64_t)i_s_1) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)255) >> (int64_t)2))]);
      } else {
        condval_3 = (bool)0;
      }
      if (condval_3) {
        tl::__sync_thread_partial(3, 128);
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)hd_shared)[(((((((i_2 >> 1) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) - 2048)])), __pack_half2(((bfloat16_t)h_fragment[(i_2 * 8)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_2 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_2 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_2 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 7)])));
        }
      }
      mbarrier[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
      tl::__sync_thread_partial(3, 128);
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        *(uint1*)(h_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)h_shared) + (((((((((i_s_1 & 1) * 4096) + ((i_3 >> 3) * 2048)) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_3 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((i_3 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 2048));
        float2 __1;
        uint1 v_ = *(uint1*)(h_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        *(float2*)(h_fragment + (i_3 * 2)) = __1;
      }
      mbarrier[((i_s_1 & 1) + 2)].arrive();
      bool condval_4;
      if (((0 <= (seq_start_idx + i_s_1)) && ((seq_start_idx + i_s_1) < cp_batch_size))) {
        condval_4 = ((bool)fallback_mask[(((((int64_t)seq_start_idx) * (int64_t)64) + (((int64_t)i_s_1) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)255) >> (int64_t)2))]);
      } else {
        condval_4 = (bool)0;
      }
      if (condval_4) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 16; ++i_4) {
          bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_5;
          if (((0 <= (seq_start_idx + i_s_1)) && ((seq_start_idx + i_s_1) < cp_batch_size))) {
            condval_5 = *(uint4*)(mt_buffer + ((((((((int64_t)seq_start_idx) * (int64_t)1048576) + (((int64_t)i_s_1) * (int64_t)1048576)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)255) >> (int64_t)2) * (int64_t)16384)) + (((int64_t)i_4) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) - (int64_t)1024));
          } else {
            condval_5 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
          }
          *(uint4*)(((bfloat16_t*)m_shared) + ((((((((i_s_1 & 1) * 16384) + (((((int)threadIdx.x) & 15) >> 3) * 8192)) + (i_4 * 512)) + (((((int)threadIdx.x) & 127) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_5;
        }
        {
          tl::GmmaDescriptor desc_a;
          tl::GmmaDescriptor desc_b;
          tl::__sync_thread_partial(3, 128);
          tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)m_shared)[((i_s_1 & 1) * 16384)])));
          tl::initialize_wgmma_descriptor<2, 0, 32>(desc_b, (&(((bfloat16_t*)hd_shared)[0])));
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_fragment + 0), 32);
          tl::warpgroup_arrive();
          tl::fence_proxy_async();
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            #pragma unroll
            for (int ki = 0; ki < 8; ++ki) {
              tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, true, 1, 1>(uint64_t(desc_a + (((((ki >> 2) * 16384) + (i_5 * 8192)) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((ki * 1024) >> 4)), ((uint32_t*)(h_fragment + (i_5 * 16))), 1);
            }
          }
          tl::warpgroup_commit_batch();
          tl::warpgroup_wait<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(h_fragment + 0), 32);
        }
      }
      if (-1 <= (seq_start_idx + i_s_1)) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 16; ++i_6) {
          if (((seq_start_idx + i_s_1) + 1) < cp_batch_size) {
            *(float2*)(cp_h0 + (((((((((((((int64_t)seq_start_idx) * (int64_t)1048576) + (((int64_t)i_s_1) * (int64_t)1048576)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)255) >> (int64_t)2) * (int64_t)16384)) + ((((int64_t)i_6) >> (int64_t)3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)1040384)) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  }
}
