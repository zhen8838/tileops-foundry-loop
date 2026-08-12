"""Blind-phase runtime twin and evidence runner for the Mamba-2 recurrence."""

import argparse
import functools
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel


_LOG2E = 1.4426950408889634


@functools.lru_cache(maxsize=8)
def build_serial_runtime(
    batch: int,
    seqlen: int,
    heads: int,
    head_dim: int,
    groups: int,
    state_dim: int,
    dtype: str,
    threads: int,
):
    """Build the graph-traceable direct recurrence, specialized by shape."""
    if groups < 1 or heads % groups:
        raise ValueError("heads must be divisible by groups")
    heads_per_group = heads // groups

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel():
        @T.prim_func
        def mamba2_serial(
            x: T.Tensor([batch, seqlen, heads, head_dim], dtype),
            dt: T.Tensor([batch, seqlen, heads], "float32"),
            A: T.Tensor([heads], "float32"),
            B: T.Tensor([batch, seqlen, groups, state_dim], dtype),
            C: T.Tensor([batch, seqlen, groups, state_dim], dtype),
            y: T.Tensor([batch, seqlen, heads, head_dim], "float32"),
            final_state: T.Tensor(
                [batch, heads, head_dim, state_dim], "float32"
            ),
        ):
            with T.Kernel(batch, heads, threads=threads) as (bid, hid):
                state = T.alloc_shared([head_dim, state_dim], "float32")
                delta_shared = T.alloc_shared([1], "float32")
                decay_shared = T.alloc_shared([1], "float32")
                y_fragment = T.alloc_fragment([head_dim], "float32")
                tx = T.get_thread_binding()
                group = hid // heads_per_group

                for p, n in T.Parallel(head_dim, state_dim):
                    state[p, n] = T.float32(0.0)
                T.sync_threads()

                for token in T.serial(seqlen):
                    if tx == 0:
                        dt_value = T.cast(dt[bid, token, hid], "float32")
                        delta = T.if_then_else(
                            dt_value > T.float32(20.0),
                            dt_value,
                            T.log(T.float32(1.0) + T.exp(dt_value)),
                        )
                        delta_shared[0] = T.max(delta, T.float32(0.0))
                        decay_shared[0] = T.exp2(
                            T.cast(A[hid], "float32")
                            * delta_shared[0]
                            * T.float32(_LOG2E)
                        )
                    T.sync_threads()

                    for p, n in T.Parallel(head_dim, state_dim):
                        state[p, n] = (
                            decay_shared[0] * state[p, n]
                            + delta_shared[0]
                            * T.cast(x[bid, token, hid, p], "float32")
                            * T.cast(B[bid, token, group, n], "float32")
                        )
                    T.sync_threads()

                    T.fill(y_fragment, T.float32(0.0))
                    for n in T.serial(state_dim):
                        for p in T.Parallel(head_dim):
                            y_fragment[p] = (
                                y_fragment[p]
                                + state[p, n]
                                * T.cast(C[bid, token, group, n], "float32")
                            )
                    for p in T.Parallel(head_dim):
                        y[bid, token, hid, p] = y_fragment[p]
                    T.sync_threads()

                for p, n in T.Parallel(head_dim, state_dim):
                    final_state[bid, hid, p, n] = state[p, n]

        return mamba2_serial

    return _kernel()


def direct_oracle(x, dt, A, B, C):
    """Independent literal FP32 recurrence; no chunked factorization."""
    batch, seqlen, heads, head_dim = x.shape
    groups, state_dim = B.shape[2:]
    heads_per_group = heads // groups
    group_index = torch.arange(heads, device=x.device) // heads_per_group
    state = torch.zeros(
        batch, heads, head_dim, state_dim, dtype=torch.float32, device=x.device
    )
    y = torch.empty(
        batch, seqlen, heads, head_dim, dtype=torch.float32, device=x.device
    )
    for token in range(seqlen):
        delta = torch.clamp(F.softplus(dt[:, token].float()), min=0.0)
        decay = torch.exp(delta * A.float().unsqueeze(0))
        B_heads = B[:, token, group_index].float()
        C_heads = C[:, token, group_index].float()
        state = (
            decay[..., None, None] * state
            + delta[..., None, None]
            * x[:, token].float()[..., None]
            * B_heads[:, :, None, :]
        )
        y[:, token] = (state * C_heads[:, :, None, :]).sum(dim=-1)
    return y, state


def make_inputs(seqlen: int, heads: int, dtype: torch.dtype):
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(1, seqlen, heads, 64, dtype=dtype, device=device) * 0.1
    dt = torch.randn(1, seqlen, heads, dtype=torch.float32, device=device) * 0.5
    A = -torch.rand(heads, dtype=torch.float32, device=device)
    B = torch.randn(1, seqlen, 1, 128, dtype=dtype, device=device) * 0.1
    C = torch.randn(1, seqlen, 1, 128, dtype=dtype, device=device) * 0.1
    return x, dt, A, B, C


def error_record(actual, expected, *, atol: float):
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "allclose": bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=1e-3)),
    }


def source_record(kernel) -> dict:
    record = {"object_type": type(kernel).__name__}
    for name in ("get_kernel_source", "get_source"):
        method = getattr(kernel, name, None)
        if callable(method):
            try:
                source = method()
            except Exception as exc:  # noqa: BLE001
                record[name] = {"error": repr(exc)}
            else:
                record[name] = source
    for name in ("lib", "rt_mod", "module", "kernel"):
        value = getattr(kernel, name, None)
        if value is not None:
            record[name] = repr(value)
    return record


def run_case(seqlen: int, heads: int, dtype: torch.dtype, *, oracle: bool):
    inputs = make_inputs(seqlen, heads, dtype)
    dtype_name = "bfloat16" if dtype is torch.bfloat16 else "float16"
    kernel = build_serial_runtime(1, seqlen, heads, 64, 1, 128, dtype_name, 128)
    y, state = kernel(*inputs)
    official_y, official_state = mamba_chunk_scan_combined(
        *inputs,
        256,
        dt_softplus=True,
        dt_bias=None,
        initial_states=None,
        return_final_states=True,
    )
    official_y = official_y.float()
    official_state = official_state.float()
    result = {
        "shape": {"batch": 1, "seqlen": seqlen, "heads": heads, "head_dim": 64, "groups": 1, "state_dim": 128},
        "dtype": str(dtype),
        "candidate": {
            "class": "blind_serial_tilelang_runtime_twin",
            "y_shape": list(y.shape),
            "y_dtype": str(y.dtype),
            "state_shape": list(state.shape),
            "state_dtype": str(state.dtype),
            "source": source_record(kernel),
        },
        "vs_official": {
            "y": error_record(y, official_y, atol=2e-2 if dtype is torch.bfloat16 else 1e-2),
            "state": error_record(state, official_state, atol=2e-2 if dtype is torch.bfloat16 else 1e-2),
        },
    }
    if oracle:
        oracle_y, oracle_state = direct_oracle(*inputs)
        result["vs_direct_oracle"] = {
            "y": error_record(y, oracle_y, atol=2e-2 if dtype is torch.bfloat16 else 1e-2),
            "state": error_record(state, oracle_state, atol=2e-2 if dtype is torch.bfloat16 else 1e-2),
        }
    return kernel, inputs, result


def quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-trials", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--junit-out", type=Path)
    parser.add_argument("--source-out", type=Path)
    args = parser.parse_args()

    _, _, small = run_case(512, 4, torch.float16, oracle=True)
    kernel, inputs, production = run_case(2048, 80, torch.bfloat16, oracle=False)
    if args.source_out is not None:
        source = getattr(kernel, "get_kernel_source", lambda: "")()
        args.source_out.write_text(source, encoding="utf-8")
        production["candidate"]["generated_source_path"] = str(args.source_out)

    trials = []
    for _ in range(args.bench_trials):
        samples = bench_kernel(kernel, args=inputs, dry_run_ms=10.0, repeat_ms=100.0)
        trials.append(
            {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p10_ms": quantile(samples, 0.1),
                "p90_ms": quantile(samples, 0.9),
                "metadata": _capture_bench_meta(),
            }
        )
    production["latency_trials"] = trials
    payload = {"small_multi_chunk": small, "production_bf16": production}
    rendered = json.dumps(payload, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")

    cases = [
        ("small_multi_chunk", small, ("vs_direct_oracle", "vs_official")),
        ("production_bf16", production, ("vs_official",)),
    ]
    suite = ET.Element("testsuite", name="mamba2_blind_candidate", tests=str(len(cases)))
    failures = 0
    for name, record, comparisons in cases:
        case = ET.SubElement(suite, "testcase", name=name)
        failed = [
            f"{comparison}.{output}"
            for comparison in comparisons
            for output in ("y", "state")
            if not record[comparison][output]["allclose"]
        ]
        if failed:
            failures += 1
            ET.SubElement(case, "failure", message=", ".join(failed)).text = rendered
    suite.set("failures", str(failures))
    if args.junit_out is not None:
        ET.ElementTree(suite).write(args.junit_out, encoding="utf-8", xml_declaration=True)
    print(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
