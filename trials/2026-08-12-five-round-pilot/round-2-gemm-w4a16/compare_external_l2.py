import statistics

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from benchmarks.ops.bench_gemm import _prepare_marlin_w4a16_baseline
from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel as Incumbent
from tileops.kernels.gemm_w4a16_r2 import GemmW4A16Kernel as Candidate
from workloads.gemm import GemmW4A16Workload


m, n, k = 1, 8192, 8192
torch.manual_seed(20260812)
w = GemmW4A16Workload(m, n, k, torch.float16)
inputs = w.gen_inputs()
expected = w.ref_program(*inputs)
implementations = {
    "candidate": (Candidate(m, n, k, torch.float16, config={"block_n": 16, "threads": 128}), inputs),
    "incumbent": (Incumbent(m, n, k, torch.float16), inputs),
}
for name, fp32 in (("marlin-fp16", False), ("marlin-fp32", True)):
    implementations[name] = _prepare_marlin_w4a16_baseline(m, n, k, fp32, *inputs)

for name, (fn, args) in implementations.items():
    actual = fn(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
    diff = (actual.float() - expected.float()).abs()
    samples = bench_kernel(fn, args=args, dry_run_ms=5.0, repeat_ms=20.0)
    print(
        name,
        _capture_bench_meta(),
        "median_ms",
        statistics.median(samples),
        "p10_ms",
        sorted(samples)[round(0.1 * (len(samples) - 1))],
        "p90_ms",
        sorted(samples)[round(0.9 * (len(samples) - 1))],
        "max_abs",
        diff.max().item(),
    )
