"""Full production-manifest correctness and native-CUPTI profile harness."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
    _prefill_auto_cp_local_chunks,
    _prefill_should_partition,
)
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload


SHAPES = [
    (4096, 16),
    (32768, 16), (65536, 16), (131072, 16),
    (32768, 32), (65536, 32), (131072, 32),
    (32768, 48), (65536, 48), (131072, 48),
    (32768, 64), (65536, 64), (131072, 64),
]


class PartitionMode:
    def __init__(
        self,
        op: GatedDeltaNetPrefillFwdOp,
        force: bool,
        max_local_chunks: int | None = None,
    ) -> None:
        self.op = op
        self.force = force
        self.max_local_chunks = max_local_chunks

    def __call__(self, *inputs):
        os.environ["TILEOPS_GDN_PREFILL_FORCE_PARTITION"] = "1" if self.force else "0"
        if self.max_local_chunks is None:
            os.environ.pop("TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS", None)
        else:
            os.environ["TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS"] = str(
                self.max_local_chunks
            )
        return self.op(*inputs)


def fla_contract(q, k, v, g, beta):
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


def sample_summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "metadata": _capture_bench_meta(),
    }


def measure_peak(fn, inputs) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = fn(*inputs)
    torch.cuda.synchronize()
    return outputs, torch.cuda.max_memory_allocated() - before


def base_auto_cp_local_chunks(num_chunks: int, num_heads: int) -> int:
    sm_count = torch.cuda.get_device_properties().multi_processor_count
    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(num_heads * num_chunks / sm_count) * 3)
    )
    if num_heads >= 64 and num_chunks >= 512:
        max_local_chunks = max(max_local_chunks, 256)
    return max(max_local_chunks, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--only", action="append", default=[],
                        help="Optional S,H,dtype selector, e.g. 4096,16,bfloat16")
    args = parser.parse_args()
    selected = set(args.only)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", buffering=1) as output_file:
        for seq_len, heads in SHAPES:
            for dtype_name in ("float16", "bfloat16"):
                selector = f"{seq_len},{heads},{dtype_name}"
                if selected and selector not in selected:
                    continue
                dtype = getattr(torch, dtype_name)
                torch.manual_seed(42)
                workload = GatedDeltaNetPrefillFwdWorkload(
                    1, heads, seq_len, 128, 128, 64, dtype, layout="bthd"
                )
                inputs = workload.gen_inputs()
                candidate_op = GatedDeltaNetPrefillFwdOp(chunk_size=64, layout="bthd")
                base_op = GatedDeltaNetPrefillFwdOp(chunk_size=64, layout="bthd")
                num_chunks = seq_len // 64
                base_max_local_chunks = base_auto_cp_local_chunks(num_chunks, heads)
                functors = {
                    "candidate": PartitionMode(candidate_op, force=False),
                    "base": PartitionMode(
                        base_op,
                        force=True,
                        max_local_chunks=base_max_local_chunks,
                    ),
                    "fla": fla_contract,
                }

                expected, fla_peak = measure_peak(fla_contract, inputs)
                errors = {
                    "fla": {
                        "max_o_abs": 0.0,
                        "max_state_abs": 0.0,
                        "peak_allocated_delta_bytes": fla_peak,
                    }
                }
                atol = rtol = 5e-2 if dtype is torch.float16 else 1e-1
                for name in ("candidate", "base"):
                    actual, peak = measure_peak(functors[name], inputs)
                    errors[name] = {
                        "max_o_abs": (actual[0].float() - expected[0].float()).abs().max().item(),
                        "max_state_abs": (
                            actual[1].float() - expected[1].float()
                        ).abs().max().item(),
                        "peak_allocated_delta_bytes": peak,
                        "kernel": type(functors[name].op.kernel).__name__,
                    }
                    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
                    torch.testing.assert_close(actual[1], expected[1], atol=atol, rtol=rtol)
                    del actual

                max_local_chunks = _prefill_auto_cp_local_chunks(num_chunks, heads)
                route = {
                    "candidate_partition": _prefill_should_partition(
                        seq_len, num_chunks, heads, max_local_chunks, False
                    ),
                    "base_partition": _prefill_should_partition(
                        seq_len, num_chunks, heads, base_max_local_chunks, True
                    ),
                    "candidate_max_local_chunks": max_local_chunks,
                    "base_max_local_chunks": base_max_local_chunks,
                }

                timings = {name: [] for name in functors}
                names = list(functors)
                for trial in range(args.trials):
                    trial_order = names if trial % 2 == 0 else names[::-1]
                    for name in trial_order:
                        timings[name].append(
                            sample_summary(bench_kernel(functors[name], args=inputs))
                        )

                record = {
                    "selector": selector,
                    "shape": {"batch": 1, "seq_len": seq_len, "heads": heads,
                              "dim_k": 128, "dim_v": 128, "chunk_size": 64},
                    "dtype": str(dtype),
                    "route": route,
                    "errors_vs_fla": errors,
                    "timings": timings,
                }
                line = json.dumps(record, sort_keys=True)
                print(line, flush=True)
                output_file.write(line + "\n")
                del expected, inputs, candidate_op, base_op, functors
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
