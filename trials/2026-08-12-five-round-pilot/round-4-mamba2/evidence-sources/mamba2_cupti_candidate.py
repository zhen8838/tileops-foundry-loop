"""Isolated CUPTI measurement for one Mamba-2 internal configuration."""

import argparse
import json
import statistics

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.manifest import load_manifest
from tileops.ops.mamba2_fwd import Mamba2FwdOp


CONFIGS = {
    "default": None,
    "scan-n64-t64-s64": (
        "_chunk_scan_op",
        {
            "block_l": 64,
            "block_p": 64,
            "block_n": 64,
            "block_s": 64,
            "threads": 64,
            "num_stages": 3,
        },
    ),
    "scan-n64-t128-s128": (
        "_chunk_scan_op",
        {
            "block_l": 64,
            "block_p": 64,
            "block_n": 64,
            "block_s": 128,
            "threads": 128,
            "num_stages": 3,
        },
    ),
    "scan-n128-t128-s64": (
        "_chunk_scan_op",
        {
            "block_l": 64,
            "block_p": 64,
            "block_n": 128,
            "block_s": 64,
            "threads": 128,
            "num_stages": 3,
        },
    ),
    "chunk-l64-t256": (
        "_chunk_state_op",
        {"block_n": 128, "block_p": 64, "block_l": 64, "threads": 256},
    ),
}


def make_inputs(row):
    torch.manual_seed(42)
    dtype = getattr(torch, row["dtypes"][0])
    return (
        torch.randn(tuple(row["x_shape"]), device="cuda", dtype=dtype) * 0.1,
        torch.randn(tuple(row["dt_shape"]), device="cuda", dtype=torch.float32) * 0.5,
        -torch.rand(tuple(row["A_shape"]), device="cuda", dtype=torch.float32),
        torch.randn(tuple(row["B_shape"]), device="cuda", dtype=dtype) * 0.1,
        torch.randn(tuple(row["C_shape"]), device="cuda", dtype=dtype) * 0.1,
    )


def output_errors(actual, expected):
    result = []
    for got, want in zip(actual, expected, strict=True):
        delta = (got.float() - want.float()).abs()
        result.append({"max_abs": float(delta.max()), "mean_abs": float(delta.mean())})
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, choices=(0, 1), required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    args = parser.parse_args()

    row = load_manifest()["Mamba2FwdOp"]["workloads"][args.row]
    inputs = make_inputs(row)

    reference_op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    reference = reference_op.forward(*inputs, return_final_states=True)
    torch.cuda.synchronize()

    op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    candidate = op.forward(*inputs, return_final_states=True)
    selection = CONFIGS[args.config]
    if selection is not None:
        component, config = selection
        getattr(op, component).kernel.config = dict(config)
        candidate = op.forward(*inputs, return_final_states=True)
    torch.cuda.synchronize()

    errors = output_errors(candidate, reference)
    samples = bench_kernel(
        lambda x, dt, A, B, C: op.forward(
            x, dt, A, B, C,
            dt_bias=None,
            initial_states=None,
            return_final_states=True,
        ),
        inputs,
    )
    ordered = sorted(samples)
    print(json.dumps({
        "label": row["label"],
        "config_name": args.config,
        "selection": selection,
        "errors": errors,
        "timing": _capture_bench_meta(),
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.1 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.9 * (len(ordered) - 1))],
    }, indent=2))


if __name__ == "__main__":
    main()
