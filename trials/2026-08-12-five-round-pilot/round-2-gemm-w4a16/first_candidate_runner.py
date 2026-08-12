import statistics

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.ops import GemmW4A16Op
from workloads.gemm import GemmW4A16Workload


torch.manual_seed(20260812)
workload = GemmW4A16Workload(1, 8192, 8192, torch.float16)
inputs = workload.gen_inputs()
op = GemmW4A16Op()
actual = op(*inputs)
expected = workload.ref_program(*inputs)
torch.cuda.synchronize()
diff = (actual.float() - expected.float()).abs()
denom = expected.float().abs().clamp_min(1e-12)
print(f"candidate_module={type(op.kernel).__module__}")
print(f"candidate_class={type(op.kernel).__name__}")
print(f"candidate_config={op.kernel.config}")
print(f"shape={tuple(actual.shape)} dtype={actual.dtype}")
print(f"max_abs={diff.max().item():.9g}")
print(f"max_rel={torch.where(diff == 0, 0.0, diff / denom).max().item():.9g}")
torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
print("allclose=True atol=0.07 rtol=0.05")

for run in range(1, 4):
    samples = bench_kernel(op, args=inputs, dry_run_ms=5.0, repeat_ms=20.0)
    ordered = sorted(samples)
    last = len(ordered) - 1
    print(
        f"run={run} timing={_capture_bench_meta()} n={len(samples)} "
        f"median_ms={statistics.median(samples):.9g} "
        f"p10_ms={ordered[round(0.1 * last)]:.9g} "
        f"p90_ms={ordered[round(0.9 * last)]:.9g}"
    )
    print("raw_ms=" + ",".join(f"{sample:.9g}" for sample in samples))
