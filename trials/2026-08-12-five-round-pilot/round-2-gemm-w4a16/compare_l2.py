import statistics

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel as Incumbent
from tileops.kernels.gemm_w4a16_r2 import GemmW4A16Kernel as Candidate
from workloads.gemm import GemmW4A16Workload


torch.manual_seed(20260812)
w = GemmW4A16Workload(1, 8192, 8192, torch.float16)
args = w.gen_inputs()
implementations = {
    "candidate": Candidate(1, 8192, 8192, torch.float16),
    "incumbent": Incumbent(1, 8192, 8192, torch.float16),
}
expected = w.ref_program(*args)
for name, fn in implementations.items():
    actual = fn(*args)
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
    samples = bench_kernel(fn, args=args, dry_run_ms=5.0, repeat_ms=20.0)
    print(
        name,
        fn.config,
        _capture_bench_meta(),
        "median_ms",
        statistics.median(samples),
        "min_ms",
        min(samples),
        "max_ms",
        max(samples),
        "max_abs",
        diff.max().item(),
    )
