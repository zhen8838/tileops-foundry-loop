"""Controlled event-timed config ablations for the primary Mamba-2 rows."""

import json
import statistics

import torch

from tileops.manifest import load_manifest
from tileops.ops.mamba2_fwd import Mamba2FwdOp


STATE_CONFIGS = {
    "state-default": {"block_d": 64, "threads": 128, "vectorize": False},
    "state-nv128": {"block_d": 128, "threads": 128, "vectorize": False},
    "state-v64": {"block_d": 64, "threads": 32, "vectorize": True},
    "state-v128": {"block_d": 128, "threads": 64, "vectorize": True},
    "state-v256": {"block_d": 256, "threads": 128, "vectorize": True},
}

SCAN_CONFIGS = {
    "scan-default": {"block_l": 64, "block_p": 64, "block_n": 64, "block_s": 64, "threads": 128, "num_stages": 3},
    "scan-n128-t128-s64": {"block_l": 64, "block_p": 64, "block_n": 128, "block_s": 64, "threads": 128, "num_stages": 3},
    "scan-n64-t64-s64": {"block_l": 64, "block_p": 64, "block_n": 64, "block_s": 64, "threads": 64, "num_stages": 3},
    "scan-n128-t64-s64": {"block_l": 64, "block_p": 64, "block_n": 128, "block_s": 64, "threads": 64, "num_stages": 3},
    "scan-n64-t128-s128": {"block_l": 64, "block_p": 64, "block_n": 64, "block_s": 128, "threads": 128, "num_stages": 3},
    "scan-n128-t128-s128": {"block_l": 64, "block_p": 64, "block_n": 128, "block_s": 128, "threads": 128, "num_stages": 3},
}

CHUNK_STATE_CONFIGS = {
    "chunk-default": {"block_n": 128, "block_p": 64, "block_l": 32, "threads": 128},
    "chunk-l64-t128": {"block_n": 128, "block_p": 64, "block_l": 64, "threads": 128},
    "chunk-l128-t128": {"block_n": 128, "block_p": 64, "block_l": 128, "threads": 128},
    "chunk-l64-t256": {"block_n": 128, "block_p": 64, "block_l": 64, "threads": 256},
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


def timed(op, inputs, repeats=20):
    for _ in range(3):
        outputs = op.forward(*inputs, return_final_states=True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = op.forward(*inputs, return_final_states=True)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return outputs, samples


def errors(actual, expected):
    result = []
    for got, want in zip(actual, expected, strict=True):
        delta = (got.float() - want.float()).abs()
        result.append({"max_abs": float(delta.max()), "mean_abs": float(delta.mean())})
    return result


@torch.inference_mode()
def run_row(row):
    inputs = make_inputs(row)
    op = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    reference, default_samples = timed(op, inputs)
    records = [{"name": "default", "samples_ms": default_samples, "median_ms": statistics.median(default_samples), "errors": errors(reference, reference)}]

    best_state = STATE_CONFIGS["state-default"]
    state_records = []
    for name, config in STATE_CONFIGS.items():
        op._state_passing_op.kernel.config = dict(config)
        outputs, samples = timed(op, inputs)
        record = {"name": name, "config": config, "samples_ms": samples, "median_ms": statistics.median(samples), "errors": errors(outputs, reference)}
        state_records.append(record)
    best_state = min(state_records, key=lambda item: item["median_ms"])["config"]
    records.extend(state_records)

    scan_records = []
    op._state_passing_op.kernel.config = dict(best_state)
    for name, config in SCAN_CONFIGS.items():
        op._chunk_scan_op.kernel.config = dict(config)
        outputs, samples = timed(op, inputs)
        record = {"name": name, "config": config, "state_config": best_state, "samples_ms": samples, "median_ms": statistics.median(samples), "errors": errors(outputs, reference)}
        scan_records.append(record)
    best_scan = min(scan_records, key=lambda item: item["median_ms"])["config"]
    records.extend(scan_records)

    op._chunk_scan_op.kernel.config = dict(best_scan)
    for name, config in CHUNK_STATE_CONFIGS.items():
        op._chunk_state_op.kernel.config = dict(config)
        outputs, samples = timed(op, inputs)
        records.append({"name": name, "config": config, "state_config": best_state, "scan_config": best_scan, "samples_ms": samples, "median_ms": statistics.median(samples), "errors": errors(outputs, reference)})

    return {"label": row["label"], "records": records}


def main():
    rows = load_manifest()["Mamba2FwdOp"]["workloads"]
    print(json.dumps([run_row(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
