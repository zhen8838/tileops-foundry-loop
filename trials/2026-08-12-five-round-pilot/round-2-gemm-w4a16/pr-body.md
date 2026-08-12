## Summary

- Add a fused M=1 W4A16 decode specialization that unpacks little-endian nibbles, applies group-128 affine dequantization with the contract's FP16 rounding point, and reduces directly to the FP16 output.
- Dispatch M=1 calls to the specialization while preserving the existing tensor-core implementation for M>1.
- Compare the candidate, exact TileOPs incumbent, and both runnable same-contract vLLM Marlin reduction modes on every primary manifest workload.

## TileFoundry Description

```python
@module(
    entry="gemm_w4a16",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", None),),
)
class GemmW4A16:
    @func
    def gemm_w4a16(
        activation: Tensor[(M, K), "f16"],
        packed_weight: Tensor[(N, K // 2), "u8"],
        weight_scale: Tensor[(N, K // 128), "f32"],
        weight_zero: Tensor[(N, K // 128), "u8"],
    ) -> Tensor[(M, N), "f16"]:
        packed_i32 = tf.cast(packed_weight, dtype="i32")
        low = tf.mod(packed_i32, 16)
        high = tf.floor_div(packed_i32, 16)
        q_i32 = tf.reshape(tf.stack(low, high, axis=-1), new_shape=(N, K))

        scale_expanded = tf.repeat_interleave(weight_scale, repeats=128, axis=1)
        zero_expanded = tf.repeat_interleave(
            tf.cast(weight_zero, dtype="f32"), repeats=128, axis=1
        )
        dequant_f32 = (tf.cast(q_i32, dtype="f32") - zero_expanded) * scale_expanded
        dequant_f16 = tf.cast(dequant_f32, dtype="f16")
        return tf.matmul(activation, tf.transpose(dequant_f16, perm=(1, 0)))
```

## Performance

Operator: `GemmW4A16Op`

| Environment | Value |
| --- | --- |
| image | ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev |
| digest | sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7 |
| gpu | NVIDIA H200 |
| driver | 595.71.05 |
| cuda | 13.2 |
| torch | 2.13.0+cu132 |
| tilelang | 0.1.11+cu132.gitafcebed1 |
| vllm | 0.27.1 |
| timer | native CUPTI |

Method: Candidate, incumbent, and both Marlin modes ran in one process on common logical inputs with identical W4A16 work and FP16 output, 25 ms warmup, 100 ms repeat policy, L2 reset/input rotation, and forward/reverse order balancing. Layout conversion, compilation, tuning, and workspace setup were excluded; CUPTI failure was fail-closed.

| Workload | M | N | K | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms) | vLLM Marlin FP16 reduce (ms) | vLLM Marlin FP32 reduce (ms) | TileOPs incumbent / candidate | vLLM Marlin FP16 reduce / candidate | vLLM Marlin FP32 reduce / candidate |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode-l2-resident-ish | 1 | 8192 | 8192 | float16 | 0.0538395 | 0.0844635 | 0.02016 | 0.020064 | 1.5688x | 0.3744x | 0.3727x |
| decode-hbm-streaming-threshold | 1 | 8192 | 16384 | float16 | 0.10112 | 0.168576 | 0.0352635 | 0.035328 | 1.6671x | 0.3487x | 0.3494x |
| decode-non-power2-low-cta | 1 | 7168 | 20480 | float16 | 0.122688 | 0.204048 | 0.0379845 | 0.037952 | 1.6631x | 0.3096x | 0.3093x |
| decode-long-k-pressure | 1 | 8192 | 81920 | float16 | 0.463393 | 0.809922 | 0.124512 | 0.124768 | 1.7478x | 0.2687x | 0.2692x |
| geometric mean |  |  |  |  | 0.132639 | 0.220247 | 0.0428212 | 0.0428024 | 1.6605x | 0.3228x | 0.3227x |

## Result And Limitations

**improvement without SOTA**

- Classification is improvement without SOTA: the candidate is 1.5688x to 1.7478x faster than the TileOPs incumbent on every primary row, with a 1.6605x geometric-mean speedup, but is 2.6834x to 3.7217x slower than the fastest passing Marlin mode and 3.1019x slower by geometric-mean latency.
- Candidate p10-to-p90 widths are 2.318%, 1.266%, 1.356%, and 0.642% of the median in table order; every incumbent win and external-baseline loss is larger than measured noise.
- The specialization applies only to M=1; M>1 continues to use the existing TileOPs tensor-core path.
- The exact TileFoundry description cannot currently be scheduled directly because the admitted TileFoundry checkout lacks unsigned-8 tensor support and dynamic RepeatInterleave shape inference, so the delivered TileLang kernel is a semantic runtime twin of that graph.
