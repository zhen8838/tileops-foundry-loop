import json
import statistics
import time

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.kernels.gemm_w4a16_r2 import GemmW4A16Kernel
from workloads.gemm import GemmW4A16Workload


SHAPES = [
    ("decode-l2-resident-ish", 1, 8192, 8192),
    ("decode-hbm-streaming-threshold", 1, 8192, 16384),
    ("decode-non-power2-low-cta", 1, 7168, 20480),
    ("decode-long-k-pressure", 1, 8192, 81920),
]
CONFIGS = [
    {"block_n": block_n, "threads": threads}
    for block_n in (4, 8, 16)
    for threads in (128, 256)
]

started = time.time()
for shape_index, (label, m, n, k) in enumerate(SHAPES):
    torch.manual_seed(20260812 + shape_index)
    workload = GemmW4A16Workload(m, n, k, torch.float16)
    inputs = workload.gen_inputs()
    expected = workload.ref_program(*inputs)
    for config in CONFIGS:
        try:
            candidate = GemmW4A16Kernel(m, n, k, torch.float16, config=config)
            actual = candidate(*inputs)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
            samples = bench_kernel(candidate, args=inputs, dry_run_ms=5.0, repeat_ms=10.0)
            ordered = sorted(samples)
            record = {
                "label": label,
                "m": m,
                "n": n,
                "k": k,
                "config": config,
                "timing": _capture_bench_meta(),
                "samples": samples,
                "median_ms": statistics.median(samples),
                "p10_ms": ordered[round(0.1 * (len(ordered) - 1))],
                "p90_ms": ordered[round(0.9 * (len(ordered) - 1))],
                "max_abs": (actual.float() - expected.float()).abs().max().item(),
            }
            print(json.dumps(record), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "label": label,
                        "config": config,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
    del expected, inputs, workload
    torch.cuda.empty_cache()

print(json.dumps({"elapsed_seconds": time.time() - started}), flush=True)
