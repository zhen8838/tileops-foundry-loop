"""One-call profiler probe for the two primary Mamba-2 manifest rows."""

import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tileops.manifest import load_manifest
from tileops.ops.mamba2_fwd import Mamba2FwdOp


def inputs_for(row):
    torch.manual_seed(42)
    dtype = getattr(torch, row["dtypes"][0])
    x = torch.randn(tuple(row["x_shape"]), device="cuda", dtype=dtype) * 0.1
    dt = torch.randn(tuple(row["dt_shape"]), device="cuda", dtype=torch.float32) * 0.5
    A = -torch.rand(tuple(row["A_shape"]), device="cuda", dtype=torch.float32)
    B = torch.randn(tuple(row["B_shape"]), device="cuda", dtype=dtype) * 0.1
    C = torch.randn(tuple(row["C_shape"]), device="cuda", dtype=dtype) * 0.1
    return x, dt, A, B, C


def configs(op):
    da = next(iter(op._da_cumsum_ops.values())).kernel
    cb = next(iter(op._cb_producer_ops.values()))._get_kernel(op.dtype)
    return {
        "da_cumsum": da.config,
        "cb_producer": cb.config,
        "chunk_state": op._chunk_state_op.kernel.config,
        "state_passing": op._state_passing_op.kernel.config,
        "chunk_scan": op._chunk_scan_op.kernel.config,
    }


def run_row(row, out_dir):
    tensors = inputs_for(row)
    op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    for _ in range(3):
        op.forward(*tensors, dt_bias=None, initial_states=None, return_final_states=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        outputs = op.forward(
            *tensors, dt_bias=None, initial_states=None, return_final_states=True
        )
        torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - before
    trace_path = out_dir / f"{row['label']}-trace.json"
    prof.export_chrome_trace(str(trace_path))

    events = []
    for event in prof.key_averages():
        cuda_us = getattr(event, "self_device_time_total", 0.0)
        if cuda_us:
            events.append(
                {
                    "key": event.key,
                    "count": event.count,
                    "self_device_time_us": cuda_us,
                    "device_memory_bytes": getattr(event, "self_device_memory_usage", 0),
                }
            )
    events.sort(key=lambda event: event["self_device_time_us"], reverse=True)
    return {
        "label": row["label"],
        "outputs": [
            {"shape": list(t.shape), "dtype": str(t.dtype)} for t in outputs
        ],
        "peak_allocated_delta_bytes": peak_delta,
        "configs": configs(op),
        "cuda_events": events,
        "trace": str(trace_path),
    }


def main():
    out_dir = Path("mamba2-profile-probe")
    out_dir.mkdir(exist_ok=True)
    rows = load_manifest()["Mamba2FwdOp"]["workloads"]
    result = [run_row(row, out_dir) for row in rows]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
