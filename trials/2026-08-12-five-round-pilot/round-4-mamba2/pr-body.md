## Summary

- Route an absent dt_bias through the existing unbiased dA-cumsum specialization and cache biased and unbiased variants independently.
- Make the shared workload and benchmark own the exact no-bias, two-output primary Mamba2 contract.
- Keep the change classified as a contract fix; it does not deliver a new performance kernel or a measurable improvement over the TileOPs incumbent.

## TileFoundry Description

```python
@module(
    entry="mamba2_step",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 512)),
)
class Mamba2Step:
    @func
    def mamba2_step(
        state: Tensor[(BATCH, HEADS, HEAD_DIM, STATE_DIM), "f32"],
        x_t: Tensor[(BATCH, HEADS, HEAD_DIM), DTYPE],
        dt_t: Tensor[(BATCH, HEADS), "f32"],
        A: Tensor[(HEADS,), "f32"],
        B_t: Tensor[(BATCH, GROUPS, STATE_DIM), DTYPE],
        C_t: Tensor[(BATCH, GROUPS, STATE_DIM), DTYPE],
    ):
        delta = tf.clamp(tf.softplus(dt_t), min_val=0.0, max_val=MAX_DT)
        decay = tf.exp(delta * tf.reshape(A, new_shape=(1, HEADS)))
        B_heads = tf.repeat_interleave(B_t, repeats=HEADS, axis=1)
        C_heads = tf.repeat_interleave(C_t, repeats=HEADS, axis=1)
        update = (
            tf.reshape(delta, new_shape=(BATCH, HEADS, 1, 1))
            * tf.reshape(tf.cast(x_t, dtype="f32"), new_shape=(BATCH, HEADS, HEAD_DIM, 1))
            * tf.reshape(
                tf.cast(B_heads, dtype="f32"),
                new_shape=(BATCH, HEADS, 1, STATE_DIM),
            )
        )
        next_state = (
            tf.reshape(decay, new_shape=(BATCH, HEADS, 1, 1)) * state + update
        )
        y_t = tf.reduce(
            next_state
            * tf.reshape(
                tf.cast(C_heads, dtype="f32"),
                new_shape=(BATCH, HEADS, 1, STATE_DIM),
            ),
            axes=(-1,),
            keepdim=False,
            kind="sum",
        )
        return y_t, next_state
```

## Performance

Operator: `Mamba2FwdOp`

| Environment | Value |
| --- | --- |
| image | ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev |
| digest | sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7 |
| gpu | NVIDIA H200 |
| driver | 595.71.05 |
| cuda | 13.2 |
| torch | 2.13.0+cu132 |
| tilelang | 0.1.11+cu132.gitafcebed1 |
| mamba_ssm | 2.3.2.post1 |
| timer | native CUPTI |

Method: Candidate, reconstructed incumbent, and mamba_ssm ran in one process on the same GPU and inputs with identical precision and two-output work, rotated implementation order, L2 reset/input shifting, and three repeated trials. Compilation was excluded, output casts required by the contract remained inside timing, and CUPTI failure was fail-closed.

| Workload | B | S | H | P | N | G | Q | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms) | mamba_ssm 2.3.2.post1 (ms) | TileOPs incumbent / candidate | mamba_ssm 2.3.2.post1 / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| mamba2-2p7b-b1-s2k | 1 | 2048 | 80 | 64 | 128 | 1 | 256 | bfloat16 | 0.723347 | 0.706002 | 0.782755 | 0.9760x | 1.0821x |
| mamba2-1p3b-b1-s8k | 1 | 8192 | 64 | 64 | 128 | 1 | 256 | float16 | 0.820499 | 0.824755 | 0.924323 | 1.0052x | 1.1265x |
| geometric mean |  |  |  |  |  |  |  |  | 0.770393 | 0.763072 | 0.850599 | 0.9905x | 1.1041x |

## Result And Limitations

**measured SOTA**

- Classification is measured SOTA only against the runnable same-contract external baseline: the candidate is faster than mamba_ssm on both primary workloads and by 1.1041x in geometric mean.
- The candidate is 2.46% slower than the TileOPs incumbent on the 2K row, 0.52% faster on the 8K row, and 0.96% slower in geometric mean; this is not an incumbent improvement.
- Across the three trial medians, candidate ranges were 0.78% and 4.15%, incumbent ranges were 26.63% and 3.11%, and mamba_ssm ranges were 2.33% and 5.36% for the 2K and 8K rows respectively; the highly variable 2K incumbent prevents a stable per-row regression claim.
- The TileFoundry module captures one recurrence step because dynamic loop-token indexing and output insertion cannot yet express the full ordered scan; the shipped fix reuses existing TileOps kernels rather than claiming that module as a new performance kernel.
- The measured-SOTA classification is specific to this CUDA 13.2 stack and mamba_ssm 2.3.2.post1.
