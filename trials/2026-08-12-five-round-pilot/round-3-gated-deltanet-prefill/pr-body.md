## Summary

- Avoid context-partition setup below 32K tokens while preserving the explicit force override.
- Select 64 local chunks for the 32K-token, 16-head partition specialization on H200.
- Compare the candidate, exact TileOps incumbent route replay, and contract-equivalent FLA on every primary manifest workload.

## TileFoundry Description

```python
@module(
    entry="delta_step",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 512)),
)
class GatedDeltaNetStep:
    @func
    def delta_step(
        state: Tensor[(B, H, DK, DV), "f32"],
        q_t: Tensor[(B, H, DK), DT],
        k_t: Tensor[(B, H, DK), DT],
        v_t: Tensor[(B, H, DV), DT],
        g_t: Tensor[(B, H), DT],
        beta_t: Tensor[(B, H), DT],
    ):
        alpha = tf.exp(tf.cast(g_t, dtype="f32"))
        qf = tf.cast(q_t, dtype="f32")
        kf = tf.cast(k_t, dtype="f32")
        vf = tf.cast(v_t, dtype="f32")
        betaf = tf.cast(beta_t, dtype="f32")
        old = tf.reshape(
            tf.matmul(tf.reshape(kf, new_shape=(B, H, 1, DK)), state),
            new_shape=(B, H, DV),
        )
        delta = tf.reshape(betaf, new_shape=(B, H, 1)) * (
            vf - tf.reshape(alpha, new_shape=(B, H, 1)) * old
        )
        next_state = (
            tf.reshape(alpha, new_shape=(B, H, 1, 1)) * state
            + tf.reshape(kf, new_shape=(B, H, DK, 1))
            * tf.reshape(delta, new_shape=(B, H, 1, DV))
        )
        out = tf.reshape(
            tf.matmul(tf.reshape(qf, new_shape=(B, H, 1, DK)), next_state),
            new_shape=(B, H, DV),
        )
        return out, next_state
```

## Performance

Operator: `GatedDeltaNetPrefillFwdOp`

| Environment | Value |
| --- | --- |
| image | ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev |
| digest | sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7 |
| gpu | NVIDIA H200 |
| driver | 595.71.05 |
| cuda | 13.2 |
| torch | 2.13.0+cu132 |
| tilelang | 0.1.11+cu132.gitafcebed1 |
| fla | 0.5.2 |
| timer | native CUPTI |

Method: Candidate, exact incumbent route replay, and FLA ran in one process on common inputs with the same precision and output/final-state contract, L2 reset, adaptive repeats, forward/reverse interleaving, and fail-closed native-CUPTI activity-window attribution; compilation was excluded. Values are medians of three interleaved trial medians except two shared-path bimodal rows, which use independent fresh-process medians of five trials.

| Workload | B | S | H | DK | DV | Chunk | Dtype | TileFoundry candidate (ms) | TileOps incumbent (ms) | FLA 0.5.2 (ms) | TileOps incumbent / candidate | FLA 0.5.2 / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| s4096-h16-fp16 | 1 | 4096 | 16 | 128 | 128 | 64 | float16 | 0.616928 | 0.816527 | 0.828863 | 1.3235x | 1.3435x |
| s4096-h16-bf16 | 1 | 4096 | 16 | 128 | 128 | 64 | bfloat16 | 0.643695 | 0.863247 | 0.838402 | 1.3411x | 1.3025x |
| s32768-h16-fp16 | 1 | 32768 | 16 | 128 | 128 | 64 | float16 | 1.04138 | 0.991553 | 1.72106 | 0.9522x | 1.6527x |
| s32768-h16-bf16 | 1 | 32768 | 16 | 128 | 128 | 64 | bfloat16 | 1.05327 | 1.0624 | 1.72312 | 1.0087x | 1.6360x |
| s65536-h16-fp16 | 1 | 65536 | 16 | 128 | 128 | 64 | float16 | 1.30615 | 1.25335 | 2.62605 | 0.9596x | 2.0105x |
| s65536-h16-bf16 | 1 | 65536 | 16 | 128 | 128 | 64 | bfloat16 | 1.26909 | 1.29792 | 2.61025 | 1.0227x | 2.0568x |
| s131072-h16-fp16 | 1 | 131072 | 16 | 128 | 128 | 64 | float16 | 1.8489 | 1.83786 | 5.01409 | 0.9940x | 2.7119x |
| s131072-h16-bf16 | 1 | 131072 | 16 | 128 | 128 | 64 | bfloat16 | 1.76559 | 1.79242 | 4.99883 | 1.0152x | 2.8313x |
| s32768-h32-fp16 | 1 | 32768 | 32 | 128 | 128 | 64 | float16 | 1.33152 | 1.26188 | 2.20289 | 0.9477x | 1.6544x |
| s32768-h32-bf16 | 1 | 32768 | 32 | 128 | 128 | 64 | bfloat16 | 1.3274 | 1.32954 | 2.21042 | 1.0016x | 1.6652x |
| s65536-h32-fp16 | 1 | 65536 | 32 | 128 | 128 | 64 | float16 | 1.72775 | 1.76714 | 3.90031 | 1.0228x | 2.2575x |
| s65536-h32-bf16 | 1 | 65536 | 32 | 128 | 128 | 64 | bfloat16 | 1.77447 | 1.74516 | 3.91097 | 0.9835x | 2.2040x |
| s131072-h32-fp16 | 1 | 131072 | 32 | 128 | 128 | 64 | float16 | 2.90817 | 2.92862 | 7.83836 | 1.0070x | 2.6953x |
| s131072-h32-bf16 | 1 | 131072 | 32 | 128 | 128 | 64 | bfloat16 | 2.92798 | 2.92158 | 7.83717 | 0.9978x | 2.6767x |
| s32768-h48-fp16 | 1 | 32768 | 48 | 128 | 128 | 64 | float16 | 1.60404 | 1.60892 | 2.8673 | 1.0030x | 1.7876x |
| s32768-h48-bf16 | 1 | 32768 | 48 | 128 | 128 | 64 | bfloat16 | 1.57741 | 1.60975 | 2.85958 | 1.0205x | 1.8128x |
| s65536-h48-fp16 | 1 | 65536 | 48 | 128 | 128 | 64 | float16 | 2.38535 | 2.37511 | 5.6201 | 0.9957x | 2.3561x |
| s65536-h48-bf16 | 1 | 65536 | 48 | 128 | 128 | 64 | bfloat16 | 2.39236 | 2.38164 | 5.60053 | 0.9955x | 2.3410x |
| s131072-h48-fp16 | 1 | 131072 | 48 | 128 | 128 | 64 | float16 | 4.0087 | 4.02045 | 11.2237 | 1.0029x | 2.7998x |
| s131072-h48-bf16 | 1 | 131072 | 48 | 128 | 128 | 64 | bfloat16 | 4.05629 | 4.05757 | 11.2043 | 1.0003x | 2.7622x |
| s32768-h64-fp16 | 1 | 32768 | 64 | 128 | 128 | 64 | float16 | 1.82181 | 1.73069 | 3.54056 | 0.9500x | 1.9434x |
| s32768-h64-bf16 | 1 | 32768 | 64 | 128 | 128 | 64 | bfloat16 | 1.69245 | 1.75229 | 3.52716 | 1.0354x | 2.0841x |
| s65536-h64-fp16 | 1 | 65536 | 64 | 128 | 128 | 64 | float16 | 2.88919 | 2.89497 | 7.11454 | 1.0020x | 2.4625x |
| s65536-h64-bf16 | 1 | 65536 | 64 | 128 | 128 | 64 | bfloat16 | 2.91934 | 2.91492 | 7.0743 | 0.9985x | 2.4233x |
| s131072-h64-fp16 | 1 | 131072 | 64 | 128 | 128 | 64 | float16 | 4.93082 | 4.91233 | 14.3031 | 0.9962x | 2.9008x |
| s131072-h64-bf16 | 1 | 131072 | 64 | 128 | 128 | 64 | bfloat16 | 4.90536 | 4.94027 | 14.3234 | 1.0071x | 2.9199x |
| geometric mean |  |  |  |  |  |  |  | 1.89298 | 1.92879 | 4.06278 | 1.0189x | 2.1462x |

## Result And Limitations

**measured SOTA**

- Classification is measured SOTA against contract-equivalent FLA 0.5.2: the candidate is faster on every primary workload and its 1.892977 ms geometric mean is lower than FLA's 4.062778 ms.
- The candidate geometric mean is 1.0189x faster than the incumbent, but it is slower on 11 of 26 individual incumbent comparisons; the durable gain is concentrated in the two 4K rows, while long-context candidate/incumbent differences mostly measure noise on the same runtime route.
- The 131072-token/64-head FP16 and BF16 rows entered shared candidate/incumbent/FLA bimodal slow states in the full run; their published values are independent fresh-process five-trial medians, while the original observations remain part of the internal evidence.
- The displayed TileFoundry module is the exact authored one-step semantic description. A dynamic scan-output insertion lowering gap prevented direct generation of the shipped prefill kernel, so the delivered optimization changes selection around the existing TileOps pipeline and is not represented as generated from this module.
- The local-chunk specialization is measured on H200; other supported architectures retain the existing correctness and fallback surface but may not share the same tuning optimum.
