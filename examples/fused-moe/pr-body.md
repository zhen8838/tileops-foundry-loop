## Summary

- Select the lower-waste BM64/stage2 fused gate/up template for sparse routed rows.
- Remove one redundant output clear while preserving the public fallback, activation, and expert-parallel behavior.
- Compare the candidate, exact TileOPs incumbent, and vLLM Triton on every primary manifest workload.

## TileFoundry Description

```python
@module(
    entry="routed_experts",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 256)),
)
class RoutedExperts:
    @func
    def routed_experts(
        hidden: Tensor[(T, H), DT],
        topk_weights: Tensor[(T, K), "f32"],
        topk_ids: Tensor[(T, K), "i32"],
        w_gate_up: ConstTensor[(E, 2 * F, H), DT],
        w_down: ConstTensor[(E, H, F), DT],
    ) -> Tensor[(T, H), DT]:
        flat_ids = tf.reshape(topk_ids, new_shape=(T * K,))
        selected_in = tf.reshape(
            tf.index_select(w_gate_up, flat_ids, dim=0),
            new_shape=(T, K, 2 * F, H),
        )
        both = tf.reshape(
            tf.matmul(selected_in, tf.reshape(hidden, new_shape=(T, 1, H, 1))),
            new_shape=(T, K, 2 * F),
        )
        inner = tf.silu(both[:, :, :F]) * both[:, :, F:]
        selected_down = tf.reshape(
            tf.index_select(w_down, flat_ids, dim=0),
            new_shape=(T, K, H, F),
        )
        down = tf.reshape(
            tf.matmul(selected_down, tf.reshape(inner, new_shape=(T, K, F, 1))),
            new_shape=(T, K, H),
        )
        weighted = tf.cast(down, dtype="f32") * tf.reshape(
            topk_weights, new_shape=(T, K, 1)
        )
        mixed = tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")
        scaled = mixed * tf.full_like(mixed, value=ROUTED_SCALING_FACTOR)
        return tf.cast(scaled, dtype=DT)
```

## Performance

Operator: `FusedMoEExpertsNopadPersistent3WGFwdOp`

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

Method: Candidate, incumbent, and vLLM ran in one process on common inputs with identical BF16 work, L2 reset/input rotation, adaptive repeats, and forward/reverse drift balancing. Compilation was excluded and CUPTI failure was fail-closed.

| Workload | T | E | K | H | F | Dtype | TileFoundry candidate (ms) | TileOPs incumbent (ms) | vLLM Triton (ms) | TileOPs incumbent / candidate | vLLM Triton / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-235b-decode | 512 | 128 | 8 | 7168 | 2048 | bfloat16 | 3.0439 | 3.1617 | 2.9422 | 1.0387x | 0.9666x |
| qwen3-235b-prefill | 4096 | 128 | 8 | 7168 | 2048 | bfloat16 | 6.0151 | 5.7425 | 6.7984 | 0.9547x | 1.1302x |
| deepseek-v3-decode | 512 | 256 | 8 | 7168 | 2048 | bfloat16 | 5.8042 | 6.1058 | 5.5988 | 1.0520x | 0.9646x |
| deepseek-v3-prefill | 4096 | 256 | 8 | 7168 | 2048 | bfloat16 | 8.21 | 8.1078 | 8.5403 | 0.9876x | 1.0402x |
| geometric mean |  |  |  |  |  |  | 5.43488 | 5.47541 | 5.56111 | 1.0075x | 1.0232x |

## Result And Limitations

**improvement without SOTA**

- Classification is improvement without SOTA: the candidate geometric mean is lower than incumbent and vLLM, but both decode rows remain slower than vLLM and both prefill rows remain slower than the incumbent.
- The public default remains unchanged; the specialization only applies where its measured contract holds.
- A general runtime vector IndexSelect lowering remains a demonstrated TileFoundry gap and is not hidden by the PR.
