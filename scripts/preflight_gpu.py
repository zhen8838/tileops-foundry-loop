#!/usr/bin/env python3
"""Run the five-round external baseline matrix and a native-CUPTI smoke."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import math
import os
import statistics
import sys
import traceback
from pathlib import Path

import torch


# This script is mounted from the loop repository, so Python would otherwise
# omit the TileOPs worktree root from sys.path.
tileops_root = Path.cwd()
if (tileops_root / "benchmarks").is_dir() and str(tileops_root) not in sys.path:
    sys.path.insert(0, str(tileops_root))


# Some baseline initialization replaces process stdout. Keep the caller's
# descriptor so the final machine-readable admission record always survives.
report_stream = os.fdopen(os.dup(sys.stdout.fileno()), "w")


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return True


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def moe_triton():
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from workloads.moe import MoeExpertsWorkload

    return fused_experts(*MoeExpertsWorkload(32, 8, 2, 256, 256, torch.bfloat16).gen_inputs())


def marlin():
    # The baseline takes the workload's own quantized tensors, so the probe runs
    # it the way the benchmark does rather than on tensors of its own making.
    from benchmarks.ops.bench_gemm import (
        GemmW4A16BenchmarkWorkload,
        _prepare_marlin_w4a16_baseline,
    )

    workload = GemmW4A16BenchmarkWorkload(1, 128, 256, torch.float16, group_size=128)
    function, inputs = _prepare_marlin_w4a16_baseline(
        1, 128, 256, True, *workload.gen_inputs()
    )
    return function(*inputs)


def fla_gated_delta():
    from benchmarks.ops.bench_gated_deltanet_prefill import _fla_prefill_fwd
    from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload

    function = _fla_prefill_fwd()
    if function is None:
        raise RuntimeError("FLA chunk_gated_delta_rule is unavailable")
    workload = GatedDeltaNetPrefillFwdWorkload(
        1, 2, 128, 128, 128, 64, torch.bfloat16, layout="bthd"
    )
    return function(*workload.gen_inputs())


def mamba_scan():
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from workloads.mamba2_e2e import Mamba2FwdWorkload

    x, dt, a, b, c, dt_bias = Mamba2FwdWorkload(
        1, 256, 4, 64, 128, 1, torch.bfloat16, 256, True
    ).gen_inputs()
    return mamba_chunk_scan_combined(
        x, dt, a, b, c, 256, dt_bias=dt_bias, dt_softplus=True
    )


def cufft():
    return torch.fft.fft(torch.randn(4, 4096, dtype=torch.complex64, device="cuda"), dim=-1)


def tilelang_fft():
    from tileops.ops import FFTC2COp

    value = torch.randn(64, dtype=torch.complex64, device="cuda")
    actual = FFTC2COp(tune=False)(value)
    torch.testing.assert_close(actual, torch.fft.fft(value), rtol=2e-3, atol=2e-3)
    return actual


def cupti_smoke():
    from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel

    left = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    # One Sample per iteration: device-busy time, wall latency, and the kernel
    # count CUPTI attributed to that call. A count of None is exactly the
    # attribution failure this probe exists to catch.
    samples = bench_kernel(torch.mm, args=(left, right))
    result = {
        "samples": len(samples),
        "median_ms": statistics.median(sample.device_busy_ms for sample in samples),
        "median_kernels": statistics.median(
            sample.n_kernels for sample in samples if sample.n_kernels is not None
        ),
        "metadata": _capture_bench_meta(),
    }
    if (
        result["metadata"].get("timing") != "cupti"
        or not math.isfinite(result["median_ms"])
        or any(sample.n_kernels is None for sample in samples)
    ):
        raise RuntimeError(f"native CUPTI attribution failed: {result}")
    return result


cases = {}
for name, function in (
    ("vllm-triton-fused-experts", moe_triton),
    ("vllm-marlin-w4a16", marlin),
    ("fla-chunk-gated-delta-rule", fla_gated_delta),
    ("mamba-ssm-chunk-scan", mamba_scan),
    ("torch-cufft", cufft),
    ("tilelang-fft", tilelang_fft),
    ("native-cupti", cupti_smoke),
):
    try:
        output = function()
        torch.cuda.synchronize()
        if not finite(output):
            raise RuntimeError("non-finite output")
        cases[name] = {"status": "ok", "detail": output if isinstance(output, dict) else None}
    except Exception as error:  # admission report must retain every failure
        cases[name] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

report = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "tilelang": version("tilelang"),
    "cupti_python": version("cupti-python"),
    "cuda_bindings": version("cuda-bindings"),
    "vllm": version("vllm"),
    "flash_linear_attention": version("flash-linear-attention"),
    "mamba_ssm": version("mamba-ssm"),
    "flashinfer": version("flashinfer-python"),
    "cases": cases,
}
print(json.dumps(report, indent=2, sort_keys=True), file=report_stream, flush=True)
report_stream.close()
raise SystemExit(any(item["status"] != "ok" for item in cases.values()))
