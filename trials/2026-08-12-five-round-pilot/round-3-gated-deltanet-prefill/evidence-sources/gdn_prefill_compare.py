"""Same-input CUPTI comparison for Gated DeltaNet prefill implementations."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.kernels.gated_deltanet.gated_deltanet_prefill_authored import (
    GatedDeltaNetPrefillAuthoredKernel,
)
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload


def fla_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    o, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0,
        initial_state=None,
        output_final_state=True,
    )
    return o, final_state.to(q.dtype)


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "metadata": _capture_bench_meta(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--include-authored", action="store_true")
    parser.add_argument("--exact-oracle", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    workload = GatedDeltaNetPrefillFwdWorkload(
        1, args.heads, args.seq_len, 128, 128, 64, dtype, layout="bthd"
    )
    inputs = workload.gen_inputs()
    base = GatedDeltaNetPrefillFwdOp(chunk_size=64, layout="bthd")
    functors = {"base": base, "fla": fla_contract}
    if args.include_authored:
        functors["authored"] = GatedDeltaNetPrefillFwdOp(
            chunk_size=64,
            layout="bthd",
            kernel_map={"GatedDeltaNetPrefillFwdKernel": GatedDeltaNetPrefillAuthoredKernel},
        )

    outputs = {name: fn(*inputs) for name, fn in functors.items()}
    torch.cuda.synchronize()
    expected = workload.ref_program(*inputs) if args.exact_oracle else outputs["fla"]
    atol = rtol = 5e-2 if dtype is torch.float16 else 1e-1
    errors = {}
    for name, actual in outputs.items():
        errors[name] = {
            "max_o_abs": (actual[0].float() - expected[0].float()).abs().max().item(),
            "max_state_abs": (actual[1].float() - expected[1].float()).abs().max().item(),
            "o_dtype": str(actual[0].dtype),
            "state_dtype": str(actual[1].dtype),
            "kernel": type(getattr(functors[name], "kernel", None)).__name__,
        }
        torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
        torch.testing.assert_close(actual[1], expected[1], atol=atol, rtol=rtol)

    timings = {name: [] for name in functors}
    order = list(functors)
    for trial in range(args.trials):
        trial_order = order if trial % 2 == 0 else order[::-1]
        for name in trial_order:
            timings[name].append(summarize(bench_kernel(functors[name], args=inputs)))

    print(json.dumps({
        "shape": {"batch": 1, "seq_len": args.seq_len, "heads": args.heads,
                  "dim_k": 128, "dim_v": 128, "chunk_size": 64},
        "dtype": str(dtype),
        "errors": errors,
        "timings": timings,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
