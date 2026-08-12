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

extern "C" __global__ void tilelang_prepare_chunk_offsets_kernel_kernel(int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, int batch_size_plus_1);
extern "C" __global__ void __launch_bounds__(32, 1) tilelang_prepare_chunk_offsets_kernel_kernel(int* __restrict__ chunk_offsets, const int* __restrict__ cu_seqlens, int batch_size_plus_1) {
  int _batch_size = 0;
  int seqlen_start_fragment[1];
  int seqlen_end_fragment[1];
  int src_buffer[1];
  extern __shared__ __align__(1024) int scan_smem[];
  _batch_size = (batch_size_plus_1 - 1);
  int condval;
  if (((((int)threadIdx.x) & 7) < batch_size_plus_1)) {
    condval = cu_seqlens[(((int)threadIdx.x) & 7)];
  } else {
    condval = 0;
  }
  seqlen_start_fragment[0] = condval;
  int condval_1;
  if ((((((int)threadIdx.x) & 7) + 1) < batch_size_plus_1)) {
    condval_1 = cu_seqlens[((((int)threadIdx.x) & 7) + 1)];
  } else {
    condval_1 = 0;
  }
  seqlen_end_fragment[0] = condval_1;
  src_buffer[0] = (seqlen_end_fragment[0] - seqlen_start_fragment[0]);
  src_buffer[0] = ((src_buffer[0] + 63) >> 6);
  if ((((int)threadIdx.x) >> 3) == 0) {
    scan_smem[(((int)threadIdx.x) & 7)] = src_buffer[0];
  }
  __syncthreads();
  tl::CumSum1D<32, false>::run((&(scan_smem[0])), (&(scan_smem[0])), 8);
  __syncthreads();
  src_buffer[0] = scan_smem[(((int)threadIdx.x) & 7)];
  chunk_offsets[0] = 0;
  if ((((int)threadIdx.x) >> 3) == 0) {
    if (((((int)threadIdx.x) & 7) + 1) < batch_size_plus_1) {
      chunk_offsets[((((int)threadIdx.x) & 7) + 1)] = src_buffer[0];
    }
  }
}
