"""Repeated common-harness CUPTI evidence for the two primary Mamba-2 rows."""

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.manifest import load_manifest
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba2_fwd import Mamba2FwdOp

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError as exc:
    raise RuntimeError("mamba_ssm is required for final Mamba-2 evidence") from exc


class BaseMamba2FwdOp(Mamba2FwdOp):
    """Reproduce the base commit's two changed internal choices."""

    def _get_da_cumsum_op(self, dtype, has_dt_bias):
        key = (dtype, True)
        if key not in self._da_cumsum_ops:
            self._da_cumsum_ops[key] = DaCumsumFwdOp(
                chunk_len=self.chunk_size,
                dtype=dtype,
                dt_softplus=self.dt_softplus,
                has_dt_bias=True,
                tune=self.tune,
            )
        return self._da_cumsum_ops[key]


def make_inputs(row):
    torch.manual_seed(20260812)
    dtype = getattr(torch, row["dtypes"][0])
    return (
        torch.randn(tuple(row["x_shape"]), device="cuda", dtype=dtype) * 0.1,
        torch.randn(tuple(row["dt_shape"]), device="cuda", dtype=torch.float32) * 0.5,
        -torch.rand(tuple(row["A_shape"]), device="cuda", dtype=torch.float32),
        torch.randn(tuple(row["B_shape"]), device="cuda", dtype=dtype) * 0.1,
        torch.randn(tuple(row["C_shape"]), device="cuda", dtype=dtype) * 0.1,
    )


def tileops_call(op, x, dt, A, B, C):
    return op.forward(
        x, dt, A, B, C,
        dt_bias=None,
        initial_states=None,
        return_final_states=True,
    )


def official_call(x, dt, A, B, C):
    y, final_state = mamba_chunk_scan_combined(
        x, dt, A, B, C,
        256,
        dt_bias=None,
        initial_states=None,
        dt_softplus=True,
        return_final_states=True,
    )
    return y.float(), final_state.float()


def errors(actual, expected):
    values = []
    for got, want in zip(actual, expected, strict=True):
        delta = (got.float() - want.float()).abs()
        values.append({
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
        })
    return values


def measure(name, fn, inputs):
    samples = bench_kernel(fn, inputs)
    ordered = sorted(samples)
    return {
        "name": name,
        "timing": _capture_bench_meta(),
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.1 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.9 * (len(ordered) - 1))],
    }


def summarize(trials):
    medians = [trial["median_ms"] for trial in trials]
    return {
        "trial_medians_ms": medians,
        "median_of_trial_medians_ms": statistics.median(medians),
        "min_trial_median_ms": min(medians),
        "max_trial_median_ms": max(medians),
        "relative_trial_range": (max(medians) - min(medians)) / statistics.median(medians),
    }


def run_row(row):
    inputs = make_inputs(row)
    candidate = Mamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)
    incumbent = BaseMamba2FwdOp(chunk_size=256, dt_softplus=True, has_initial_states=False)

    candidate_outputs = tileops_call(candidate, *inputs)
    incumbent_outputs = tileops_call(incumbent, *inputs)
    incumbent._chunk_state_op.kernel.config = {
        "block_n": 128,
        "block_p": 64,
        "block_l": 32,
        "threads": 128,
    }
    incumbent_outputs = tileops_call(incumbent, *inputs)
    official_outputs = official_call(*inputs)
    torch.cuda.synchronize()

    callables = {
        "candidate": lambda *args: tileops_call(candidate, *args),
        "base_incumbent": lambda *args: tileops_call(incumbent, *args),
        "official_2.3.2.post1": official_call,
    }
    orders = [
        ("base_incumbent", "candidate", "official_2.3.2.post1"),
        ("official_2.3.2.post1", "base_incumbent", "candidate"),
        ("candidate", "official_2.3.2.post1", "base_incumbent"),
    ]
    trials = {name: [] for name in callables}
    execution_order = []
    for trial_index, order in enumerate(orders):
        for name in order:
            result = measure(name, callables[name], inputs)
            result["trial"] = trial_index
            trials[name].append(result)
            execution_order.append({"trial": trial_index, "name": name})

    return {
        "label": row["label"],
        "dtype": row["dtypes"][0],
        "candidate_config": candidate._chunk_state_op.kernel.config,
        "base_config": incumbent._chunk_state_op.kernel.config,
        "candidate_da_has_bias": next(
            iter(candidate._da_cumsum_ops.values())
        ).has_dt_bias,
        "base_da_has_bias": next(
            iter(incumbent._da_cumsum_ops.values())
        ).has_dt_bias,
        "outputs": [
            {"shape": list(value.shape), "dtype": str(value.dtype)}
            for value in candidate_outputs
        ],
        "errors_vs_official": errors(candidate_outputs, official_outputs),
        "errors_vs_base": errors(candidate_outputs, incumbent_outputs),
        "execution_order": execution_order,
        "trials": trials,
        "summary": {name: summarize(values) for name, values in trials.items()},
    }


def write_junit(path, rows):
    suite = ET.Element(
        "testsuite",
        name="mamba2-final-evidence",
        tests=str(len(rows)),
        failures="0",
        errors="0",
    )
    for row in rows:
        case = ET.SubElement(suite, "testcase", name=row["label"])
        properties = ET.SubElement(case, "properties")
        for output_index, error in enumerate(row["errors_vs_official"]):
            ET.SubElement(
                properties,
                "property",
                name=f"output_{output_index}_max_abs",
                value=str(error["max_abs"]),
            )
        for name, summary in row["summary"].items():
            ET.SubElement(
                properties,
                "property",
                name=f"{name}_median_ms",
                value=str(summary["median_of_trial_medians_ms"]),
            )
        tolerance = 0.01 if row["dtype"] == "float16" else 0.02
        if any(error["max_abs"] > tolerance for error in row["errors_vs_official"]):
            suite.set("failures", str(int(suite.get("failures")) + 1))
            failure = ET.SubElement(case, "failure", message="correctness tolerance exceeded")
            failure.text = json.dumps(row["errors_vs_official"])
    ET.ElementTree(suite).write(path, encoding="unicode", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = load_manifest()["Mamba2FwdOp"]["workloads"]
    rows = [run_row(row) for row in manifest_rows]
    candidate_geo = math.prod(
        row["summary"]["candidate"]["median_of_trial_medians_ms"] for row in rows
    ) ** (1 / len(rows))
    base_geo = math.prod(
        row["summary"]["base_incumbent"]["median_of_trial_medians_ms"] for row in rows
    ) ** (1 / len(rows))
    official_geo = math.prod(
        row["summary"]["official_2.3.2.post1"]["median_of_trial_medians_ms"] for row in rows
    ) ** (1 / len(rows))
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "protocol": {
            "trials": 3,
            "orders": "rotated",
            "timing": "native CUPTI via benchmarks.benchmark_base.bench_kernel",
            "same_process_inputs": True,
            "official_cast_inside_timing": True,
        },
        "rows": rows,
        "geomean_ms": {
            "candidate": candidate_geo,
            "base_incumbent": base_geo,
            "official_2.3.2.post1": official_geo,
        },
        "geomean_speedup": {
            "vs_base": base_geo / candidate_geo,
            "vs_official": official_geo / candidate_geo,
        },
    }
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    write_junit(args.junit, rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
