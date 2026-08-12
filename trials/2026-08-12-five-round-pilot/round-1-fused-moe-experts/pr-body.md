## Summary

- select the lower-waste BM64/stage2 fused gate/up template when routed rows per expert are at most 16
- remove a redundant fused gate/up output memset; all consumed rows are fully overwritten
- compare fused activation, the default incumbent, and vLLM Triton in one drift-balanced manifest benchmark
- move the BF16/SILU FP32 reference into `MoeExpertsWorkload` and prove benchmark rows build the fused kernel

## H200 performance

Revalidated on the official CUDA 13.2 image
`ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev` at digest
`sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`.
The stack is PyTorch 2.13.0+cu132, TileLang
0.1.11+cu132.gitafcebed1, vLLM 0.27.1, and cupti-python 13.2.0 on the
same H200/595.71.05 driver.

BF16, K=8, H=7168, F=2048. Candidate/incumbent/vLLM use the same process,
inputs, L2/input-rotation policy, and forward/reverse drift balancing. Native
CUPTI attributed all 12 rows; CUDA-event fallback was disabled.

| T | E | fused candidate | default incumbent | vLLM Triton | incumbent / candidate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 128 | 3.0439 ms | 3.1617 ms | 2.9422 ms | 1.0387x |
| 4096 | 128 | 6.0151 ms | 5.7425 ms | 6.7984 ms | 0.9547x |
| 512 | 256 | 5.8042 ms | 6.1058 ms | 5.5988 ms | 1.0520x |
| 4096 | 256 | 8.2100 ms | 8.1078 ms | 8.5403 ms | 0.9876x |
| geometric mean | | 5.4349 ms | 5.4754 ms | 5.5611 ms | 1.0075x |

This remains an improvement without a SOTA claim: both decode rows remain
slower than vLLM, and both prefill rows are slower than the default unfused
pipeline. The public default is unchanged.

Against the prior CUDA 12.9 report, candidate and incumbent geometric means
move only +0.24% and -0.14%. vLLM Qwen prefill improved materially from
7.3834 to 6.7984 ms; an isolated same-contract CUPTI confirmation measured
6.9887 ms. vLLM uses the same BF16 Triton tile defaults in 0.19.1 and 0.27.1,
and its new TD/swap-AB paths are inactive here. Because CUDA, PyTorch, Triton,
vLLM, and the timer all changed, the result is recorded as a stack-level
external-baseline improvement rather than assigned to one component.

For the changed 256-expert decode configuration, a full-op A/B measured 5.8068 ms with BM64/stage2 versus 5.8863 ms with the old BM128/stage3 default; vLLM moved only 5.6412 to 5.6393 ms between those runs.

## Validation

- Full combined supported surface, including nightly GELU parity: 189 passed
- `TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0 ... bench_fused_moe_experts.py`: 4 passed, all rows `timing=cupti`
- Official runner-image and runtime-stack verifiers: passed
- `python -m ruff check ...`: passed
- CUDA 13.2 CI GPU smoke: 2,915 passed / 14 skipped; diff-scoped full run: 33 passed; zero failures
- All applicable PR checks are green; GitHub reports `mergeable=MERGEABLE`

`scripts/test_node_delta.py`: 33 -> 34 nodes (+1, 3.0%). The one new node exercises the sparse configuration with empty experts and a partial M tile.

The PR source needed no CUDA 13.2 fix. Its single commit was rebased unchanged
onto the official runner migration commit `0d75579`.

The remaining merge gate is the protected branch's required approving human
review of the last push; there are no review comments or requested changes.
