"""Correctness evidence runner for the blind Gated DeltaNet prefill candidate."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.kernels.gated_deltanet.gated_deltanet_prefill_authored import (
    GatedDeltaNetPrefillAuthoredKernel,
)
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--bench-trials", type=int, default=0)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    workload = GatedDeltaNetPrefillFwdWorkload(
        1, args.heads, args.seq_len, 128, 128, 64, dtype, layout="bthd"
    )
    inputs = workload.gen_inputs()
    expected_o, expected_state = workload.ref_program(*inputs)
    op = GatedDeltaNetPrefillFwdOp(
        chunk_size=64,
        layout="bthd",
        kernel_map={"GatedDeltaNetPrefillFwdKernel": GatedDeltaNetPrefillAuthoredKernel},
    )
    actual_o, actual_state = op(*inputs)
    torch.cuda.synchronize()

    atol = rtol = 5e-2 if dtype is torch.float16 else 1e-1
    o_error = (actual_o.float() - expected_o.float()).abs().max().item()
    state_error = (actual_state.float() - expected_state.float()).abs().max().item()
    torch.testing.assert_close(actual_o, expected_o, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_state, expected_state, atol=atol, rtol=rtol)
    result = {
        "seq_len": args.seq_len,
        "heads": args.heads,
        "dtype": str(dtype),
        "kernel": type(op.kernel).__name__,
        "o_shape": list(actual_o.shape),
        "o_dtype": str(actual_o.dtype),
        "state_shape": list(actual_state.shape),
        "state_dtype": str(actual_state.dtype),
        "max_o_abs": o_error,
        "max_state_abs": state_error,
        "atol": atol,
        "rtol": rtol,
        "passed": True,
    }
    if args.bench_trials:
        trials = []
        for trial in range(args.bench_trials):
            samples = bench_kernel(op, args=inputs)
            ordered = sorted(samples)
            trials.append({
                "trial": trial,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
                "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
                "metadata": _capture_bench_meta(),
            })
        result["benchmark_trials"] = trials
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
