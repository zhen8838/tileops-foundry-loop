## Summary

- Consume contiguous complex inputs through a metadata-only interleaved real view, removing the two public-wrapper real and imaginary copy kernels.
- Reuse the stage-indexed twiddle lookup table in shared-memory stages and select the measured 256-thread N=4096 configuration.
- Preserve the public split-input kernel and custom-kernel contracts while comparing the candidate, exact TileOPs incumbent, and PyTorch cuFFT on every primary workload.

## TileFoundry Description

```python
@module(
    entry="dft_pair_f32",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 256)),
)
class FFTPairF32:
    @func
    def complex_butterfly(
        even_r: Tensor[(BATCH, HALF), "f32"],
        even_i: Tensor[(BATCH, HALF), "f32"],
        odd_r: Tensor[(BATCH, HALF), "f32"],
        odd_i: Tensor[(BATCH, HALF), "f32"],
        tw_r: Tensor[(HALF,), "f32"],
        tw_i: Tensor[(HALF,), "f32"],
    ):
        prod_r = tf.sub(tf.mul(odd_r, tw_r), tf.mul(odd_i, tw_i))
        prod_i = tf.add(tf.mul(odd_r, tw_i), tf.mul(odd_i, tw_r))
        out_r = tf.concat(tf.add(even_r, prod_r), tf.sub(even_r, prod_r), axis=1)
        out_i = tf.concat(tf.add(even_i, prod_i), tf.sub(even_i, prod_i), axis=1)
        return out_r, out_i

    @func
    def dft_pair_f32(
        x_r: Tensor[(BATCH, N), "f32"],
        x_i: Tensor[(BATCH, N), "f32"],
        w_r: Tensor[(N, N), "f32"],
        w_i: Tensor[(N, N), "f32"],
    ):
        y_r = tf.sub(tf.matmul(x_r, w_r), tf.matmul(x_i, w_i))
        y_i = tf.add(tf.matmul(x_r, w_i), tf.matmul(x_i, w_r))
        return y_r, y_i
```

## Performance

Operator: `FFTC2COp`

| Environment | Value |
| --- | --- |
| image | ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev |
| digest | sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7 |
| gpu | NVIDIA H200 |
| driver | 595.71.05 |
| cuda | 13.2 |
| torch | 2.13.0+cu132 |
| tilelang | 0.1.11+cu132.gitafcebed1 |
| cupti-python | 13.2.0 |
| timer | native CUPTI |

Method: Candidate, fixed-base incumbent, and warmed PyTorch cuFFT ran in one process on identical inputs with the same precision and public-call work. The current benchmark contract used native-CUPTI call-window attribution, L2 reset, forward/reverse drift balancing within each comparison, and three rotated three-way comparisons with 200 samples per implementation per trial. Compilation, tuning, lookup-table creation, and cuFFT plan creation were excluded; each value is the median of three trial medians and CUPTI failure was fail-closed.

| Workload | N | B | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms) | PyTorch cuFFT (ms) | TileOPs incumbent / candidate | PyTorch cuFFT / candidate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| fft-4k-c64-unbatched | 4096 | 1 | complex64 | 0.00664 | 0.11816 | 0.004608 | 17.7952x | 0.6940x |
| fft-4k-c64-b64 | 4096 | 64 | complex64 | 0.009792 | 0.107776 | 0.004896 | 11.0065x | 0.5000x |
| fft-4k-c128-b64 | 4096 | 64 | complex128 | 0.012928 | 0.120848 | 0.007392 | 9.3478x | 0.5718x |
| geometric mean |  |  |  | 0.0094375 | 0.115454 | 0.00550434 | 12.2336x | 0.5832x |

## Result And Limitations

**improvement without SOTA**

- Classification is improvement without SOTA: the candidate is 17.7952x, 11.0065x, and 9.3478x faster than the TileOPs incumbent, but 1.4410x, 2.0000x, and 1.7489x slower than cuFFT on the three rows respectively.
- The candidate, incumbent, and cuFFT trial-median spans are 1.21% / 14.99% / 0.67%, 0.33% / 3.31% / 0.65%, and 0.25% / 10.01% / 0.00% across the three rows; even the favorable endpoints do not change either conclusion.
- The candidate still uses two FFT kernels and a split temporary global-memory round trip, while cuFFT uses one specialized kernel; launch overhead dominates the unbatched gap and the extra global transfer dominates the batched gaps.
