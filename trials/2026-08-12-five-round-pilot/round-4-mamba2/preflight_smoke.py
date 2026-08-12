from __future__ import annotations

import inspect
import json
import statistics

import torch

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from tileops.manifest import load_manifest
from tileops.ops.mamba2_fwd import Mamba2FwdOp


def timed_ms(fn, *, warmup: int = 3, repeats: int = 10) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_row(row: dict[str, object]) -> dict[str, object]:
    torch.manual_seed(42)
    x_shape = tuple(row["x_shape"])
    dt_shape = tuple(row["dt_shape"])
    b_shape = tuple(row["B_shape"])
    dtype = getattr(torch, row["dtypes"][0])
    device = "cuda"

    x = torch.randn(x_shape, dtype=dtype, device=device) * 0.1
    dt = torch.randn(dt_shape, dtype=torch.float32, device=device) * 0.5
    A = -torch.rand(tuple(row["A_shape"]), dtype=torch.float32, device=device)
    B = torch.randn(b_shape, dtype=dtype, device=device) * 0.1
    C = torch.randn(tuple(row["C_shape"]), dtype=dtype, device=device) * 0.1

    def official_native():
        return mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            256,
            dt_softplus=True,
            return_final_states=True,
        )

    def official_contract():
        y, state = official_native()
        return y.float(), state.float()

    incumbent = Mamba2FwdOp(
        chunk_size=256,
        dt_softplus=True,
        has_initial_states=False,
    )

    def incumbent_contract():
        return incumbent.forward(
            x,
            dt,
            A,
            B,
            C,
            return_final_states=True,
        )

    y_native, state_native = official_native()
    y_base, state_base = official_contract()
    y_inc, state_inc = incumbent_contract()
    torch.cuda.synchronize()

    y_delta = (y_inc.float() - y_base).abs()
    state_delta = (state_inc.float() - state_base).abs()
    baseline_samples = timed_ms(official_contract)
    incumbent_samples = timed_ms(incumbent_contract)

    return {
        "label": row["label"],
        "dtype": str(dtype),
        "official_native": {
            "y_shape": list(y_native.shape),
            "y_dtype": str(y_native.dtype),
            "state_shape": list(state_native.shape),
            "state_dtype": str(state_native.dtype),
            "finite": bool(torch.isfinite(y_native).all() and torch.isfinite(state_native).all()),
        },
        "contract_outputs": {
            "y_dtype": str(y_base.dtype),
            "state_dtype": str(state_base.dtype),
        },
        "incumbent_vs_official": {
            "y_max_abs": float(y_delta.max()),
            "y_mean_abs": float(y_delta.mean()),
            "y_allclose_atol_2e_2_rtol_1e_3": bool(
                torch.allclose(y_inc.float(), y_base, atol=2e-2, rtol=1e-3)
            ),
            "state_max_abs": float(state_delta.max()),
            "state_mean_abs": float(state_delta.mean()),
            "state_allclose_atol_2e_2_rtol_1e_3": bool(
                torch.allclose(state_inc.float(), state_base, atol=2e-2, rtol=1e-3)
            ),
        },
        "latency_ms": {
            "official_contract_samples": baseline_samples,
            "official_contract_median": statistics.median(baseline_samples),
            "official_contract_p10": quantile(baseline_samples, 0.1),
            "official_contract_p90": quantile(baseline_samples, 0.9),
            "incumbent_samples": incumbent_samples,
            "incumbent_median": statistics.median(incumbent_samples),
            "incumbent_p10": quantile(incumbent_samples, 0.1),
            "incumbent_p90": quantile(incumbent_samples, 0.9),
        },
    }


def main() -> None:
    entry = load_manifest()["Mamba2FwdOp"]
    print(
        json.dumps(
            {
                "baseline_signature": str(inspect.signature(mamba_chunk_scan_combined)),
                "manifest_entry": entry,
                "rows": [run_row(row) for row in entry["workloads"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
