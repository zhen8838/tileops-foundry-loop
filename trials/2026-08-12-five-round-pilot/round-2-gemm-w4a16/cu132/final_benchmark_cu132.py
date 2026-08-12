import json
import os
import statistics
import time

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from benchmarks.ops.bench_gemm import _prepare_marlin_w4a16_baseline
from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel, _gemm_w4a16_kernel
from workloads.gemm import GemmW4A16Workload


SHAPES = list(enumerate([
    ("decode-l2-resident-ish", 1, 8192, 8192),
    ("decode-hbm-streaming-threshold", 1, 8192, 16384),
    ("decode-non-power2-low-cta", 1, 7168, 20480),
    ("decode-long-k-pressure", 1, 8192, 81920),
]))
if only_label := os.environ.get("TILEOPS_BENCH_LABEL"):
    SHAPES = [item for item in SHAPES if item[1][0] == only_label]
    if not SHAPES:
        raise ValueError(f"unknown TILEOPS_BENCH_LABEL={only_label!r}")
INCUMBENT_CONFIG = {
    "block_m": 64,
    "block_n": 64,
    "block_k": 64,
    "num_stages": 2,
    "threads": 128,
}


started = time.time()
for shape_index, (label, m, n, k) in SHAPES:
    torch.manual_seed(20260812 + shape_index)
    workload = GemmW4A16Workload(m, n, k, torch.float16)
    inputs = workload.gen_inputs()
    expected = workload.ref_program(*inputs)

    candidate = GemmW4A16Kernel(m, n, k, torch.float16)
    incumbent = _gemm_w4a16_kernel(m, n, k, "float16")(**INCUMBENT_CONFIG)
    functions = {
        "candidate": (candidate, inputs),
        "base-incumbent": (incumbent, inputs),
        "marlin-fp16": _prepare_marlin_w4a16_baseline(m, n, k, False, *inputs),
        "marlin-fp32": _prepare_marlin_w4a16_baseline(m, n, k, True, *inputs),
        "torch-oracle": (workload.ref_program, inputs),
    }
    correctness = {}
    for tag, (fn, args) in functions.items():
        actual = fn(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
        diff = (actual.float() - expected.float()).abs()
        rel = diff / expected.float().abs().clamp_min(1e-12)
        correctness[tag] = {
            "max_abs": diff.max().item(),
            "max_rel": rel.max().item(),
        }
    print(
        json.dumps(
            {
                "kind": "correctness",
                "label": label,
                "shape": [m, n, k],
                "candidate_delegate_module": type(candidate.decode_kernel).__module__,
                "candidate_config": candidate.config,
                "errors": correctness,
            }
        ),
        flush=True,
    )

    samples_by_tag = {tag: [] for tag in functions}
    pass_medians = {tag: [] for tag in functions}
    tags = list(functions)
    for pass_index, order in enumerate((tags, tags[::-1]), start=1):
        for tag in order:
            fn, args = functions[tag]
            samples = bench_kernel(fn, args=args)
            samples_by_tag[tag].extend(samples)
            pass_medians[tag].append(statistics.median(samples))
            print(
                json.dumps(
                    {
                        "kind": "samples",
                        "label": label,
                        "shape": [m, n, k],
                        "pass": pass_index,
                        "tag": tag,
                        "timing": _capture_bench_meta(),
                        "samples_ms": samples,
                    }
                ),
                flush=True,
            )

    for tag, samples in samples_by_tag.items():
        ordered = sorted(samples)
        print(
            json.dumps(
                {
                    "kind": "summary",
                    "label": label,
                    "shape": [m, n, k],
                    "tag": tag,
                    "n": len(samples),
                    "median_ms": statistics.median(samples),
                    "p10_ms": ordered[round(0.1 * (len(ordered) - 1))],
                    "p90_ms": ordered[round(0.9 * (len(ordered) - 1))],
                    "pass_medians_ms": pass_medians[tag],
                }
            ),
            flush=True,
        )

    del expected, functions, inputs, workload, candidate, incumbent
    torch.cuda.empty_cache()

print(json.dumps({"kind": "run", "elapsed_seconds": time.time() - started}), flush=True)
