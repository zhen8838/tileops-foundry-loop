# CUDA 12.9 to CUDA 13.2 raw median comparison

All latency values are milliseconds. Delta is `(cu132 / cu129 - 1) * 100`.
CUDA 12.9 used CUDA events because cupti-python was absent. CUDA 13.2 used
native CUPTI with fallback disabled, so sub-percent deltas are not causal
migration claims.

| workload | implementation | CUDA 12.9 | CUDA 13.2 | delta |
| --- | --- | ---: | ---: | ---: |
| qwen3-235b-decode | tileops-fused-act | 3.0369 | 3.0439 | +0.23% |
| qwen3-235b-decode | base-incumbent | 3.1552 | 3.1617 | +0.21% |
| qwen3-235b-decode | vllm-triton | 3.0022 | 2.9422 | -2.00% |
| qwen3-235b-prefill | tileops-fused-act | 6.0420 | 6.0151 | -0.45% |
| qwen3-235b-prefill | base-incumbent | 5.7479 | 5.7425 | -0.09% |
| qwen3-235b-prefill | vllm-triton | 7.3834 | 6.7984 | -7.92% |
| deepseek-v3-decode | tileops-fused-act | 5.7930 | 5.8042 | +0.19% |
| deepseek-v3-decode | base-incumbent | 6.0943 | 6.1058 | +0.19% |
| deepseek-v3-decode | vllm-triton | 5.6403 | 5.5988 | -0.74% |
| deepseek-v3-prefill | tileops-fused-act | 8.1298 | 8.2100 | +0.99% |
| deepseek-v3-prefill | base-incumbent | 8.1776 | 8.1078 | -0.85% |
| deepseek-v3-prefill | vllm-triton | 8.6845 | 8.5403 | -1.66% |
| geometric mean | tileops-fused-act | 5.4219 | 5.4349 | +0.24% |
| geometric mean | base-incumbent | 5.4830 | 5.4754 | -0.14% |
| geometric mean | vllm-triton | 5.7403 | 5.5611 | -3.12% |

Qwen prefill confirmation on CUDA 13.2: candidate 6.0258, incumbent 5.7274,
vLLM 6.9887; every row used native CUPTI.
