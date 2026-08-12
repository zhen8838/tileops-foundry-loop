# ❌ TileOPs GPU Smoke Report

> `b94331b`

## Summary

| | |
|---|---|
| **Correctness** | ❌ 2 failed |
| **gpu-smoke target** | `smoke, full (diff-scoped changed test files)` |
| **Gpu-smoke ops number** | 39 (37 passed, 0 skipped) |
| **Gpu-smoke Failures** | 2 |
| **Failures ops** | GemmOp |

## Failure Details

| Op | Testcase | Failure Reason |
|:---|:---------|:---------------|
| GemmOp | test_gemm[full-fp16-tuned-wide] | AssertionError: Tensor-likes are not close! Mismatched elements: 11 / 7168 (0.2%) Greatest absolute difference: 0.00390625 at index (0, 3966) (up to 0.001 al... |
| GemmOp | test_gemm[full-fp16-tuned-thin-n] | AssertionError: Tensor-likes are not close! Mismatched elements: 15 / 7168 (0.2%) Greatest absolute difference: 0.00390625 at index (1401, 0) (up to 0.001 al... |
