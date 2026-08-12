"""Final same-process FFT correctness, CUPTI timing, and launch attribution."""

import argparse
import importlib.util
import json
import math
import statistics
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from benchmarks.benchmark_base import (
    _capture_bench_meta,
    _ordered_trace_kernels,
    bench_kernel,
    collect_discovery,
)
from tileops.ops import FFTC2COp


ROWS = (
    ("fft-4k-c64-unbatched", (4096,), torch.complex64),
    ("fft-4k-c64-b64", (64, 4096), torch.complex64),
    ("fft-4k-c128-b64", (64, 4096), torch.complex128),
)


def load_base_kernel(path: Path):
    """Load the exact base source with only its custom-op symbol renamed."""
    source = path.read_text(encoding="utf-8")
    source = source.replace(
        '"top::fft_c2c_wrapped_kernel"', '"top::fft_c2c_base_wrapped_kernel"'
    ).replace("_fft_c2c_wrapped_kernel", "_fft_c2c_base_wrapped_kernel")
    name = "tileops.kernels.fft_base_snapshot"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "tileops.kernels"
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class BaseRunner:
    def __init__(self, module, shape, dtype):
        self.shape = shape
        self.n = shape[-1]
        self.batch = math.prod(shape[:-1]) if len(shape) > 1 else 1
        self.kernel = module.FFTC2CKernel(self.n, self.batch, dtype, tune=False)
        self.lut_real, self.lut_imag = FFTC2COp._build_lut(
            self.n, dtype, torch.device("cuda")
        )

    def __call__(self, x):
        x_real = x.real.contiguous().reshape(self.batch, self.n)
        x_imag = x.imag.contiguous().reshape(self.batch, self.n)
        pair = self.kernel(x_real, x_imag, self.lut_real, self.lut_imag)
        return torch.view_as_complex(pair).reshape(self.shape)


def compare(actual, expected, tolerance):
    delta = (actual - expected).abs()
    passed = True
    assertion = None
    try:
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance
        )
    except AssertionError as exc:
        passed = False
        assertion = str(exc)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_complex_abs": float(delta.max()),
        "mean_complex_abs": float(delta.mean()),
        "assert_close": passed,
        "assertion": assertion,
    }


def summarize_samples(samples):
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.1 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.9 * (len(ordered) - 1))],
        "metadata": _capture_bench_meta(),
    }


def launch_trace(fn, x):
    _, operator_traces = collect_discovery(
        lambda _: fn(x), 3, lambda _: None
    )
    result = []
    for records in operator_traces:
        kernels = _ordered_trace_kernels(records)
        result.append(
            [
                {
                    **record,
                    "duration_us": (record["end_ns"] - record["start_ns"]) / 1000.0,
                }
                for record in kernels
            ]
        )
    return result


def memory_record(fn, x):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    output = fn(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    output_bytes = output.numel() * output.element_size()
    return {
        "allocated_before_bytes": before,
        "peak_increment_bytes": peak - before,
        "output_bytes": output_bytes,
        "temporary_upper_bound_bytes": max(0, peak - before - output_bytes),
    }


def kernel_source(kernel):
    compiled = kernel.kernel(
        kernel.config["block_size"], kernel.config["threads"]
    )
    return compiled.get_kernel_source()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    base_module = load_base_kernel(args.base_source)
    payload = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "base_source": str(args.base_source),
        "base_transform": "custom-op symbol rename only",
        "plan_cache_initial": {
            "size": torch.backends.cuda.cufft_plan_cache.size,
            "max_size": torch.backends.cuda.cufft_plan_cache.max_size,
        },
        "rows": {},
    }
    suite = ET.Element(
        "testsuite", name="fft_final_evidence", tests=str(len(ROWS) * 3)
    )
    failures = 0

    for row_index, (label, shape, dtype) in enumerate(ROWS):
        torch.manual_seed(101 + row_index)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        candidate = FFTC2COp(tune=False)
        base = BaseRunner(base_module, shape, dtype)
        reference = lambda value: torch.fft.fft(value, dim=-1)

        candidate(x)
        base(x)
        reference(x)
        torch.cuda.synchronize()
        plan_size = torch.backends.cuda.cufft_plan_cache.size
        expected = reference(x)
        tolerance = 1e-4 if dtype is torch.complex64 else 1e-8
        implementations = {
            "candidate": candidate,
            "base-incumbent": base,
            "torch-cufft": reference,
        }
        row = {
            "shape": list(shape),
            "dtype": str(dtype),
            "cufft_plan_cache_size_after_warmup": plan_size,
            "config": {
                "candidate": candidate.kernel.config,
                "base-incumbent": base.kernel.config,
            },
            "correctness": {},
            "trials": {name: [] for name in implementations},
            "launch_traces": {},
            "memory": {},
        }
        for name, fn in implementations.items():
            actual = fn(x)
            correctness = compare(actual, expected, tolerance)
            row["correctness"][name] = correctness
            case = ET.SubElement(suite, "testcase", name=f"{label}-{name}")
            if not correctness["assert_close"]:
                failures += 1
                ET.SubElement(case, "failure", message="assert_close failed").text = correctness[
                    "assertion"
                ]
            row["launch_traces"][name] = launch_trace(fn, x)
            row["memory"][name] = memory_record(fn, x)

        tags = list(implementations)
        for trial in range(args.trials):
            offset = trial % len(tags)
            order = tags[offset:] + tags[:offset]
            for name in order:
                samples = bench_kernel(
                    implementations[name],
                    args=(x,),
                    dry_run_ms=10.0,
                    repeat_ms=50.0,
                )
                record = summarize_samples(samples)
                record["trial"] = trial
                record["order"] = order
                row["trials"][name].append(record)
                if record["metadata"].get("timing") != "cupti":
                    failures += 1
                    case = ET.SubElement(
                        suite, "testcase", name=f"{label}-{name}-timing-{trial}"
                    )
                    ET.SubElement(case, "failure", message="timing is not CUPTI")

        candidate_source = args.artifact_dir / f"{label}-candidate.generated.cu"
        base_source = args.artifact_dir / f"{label}-base.generated.cu"
        candidate_source.write_text(
            kernel_source(candidate.kernel), encoding="utf-8"
        )
        base_source.write_text(kernel_source(base.kernel), encoding="utf-8")
        row["generated_source"] = {
            "candidate": str(candidate_source),
            "base-incumbent": str(base_source),
        }
        payload["rows"][label] = row
        print(label, {name: [t["median_ms"] for t in trials] for name, trials in row["trials"].items()})

    payload["plan_cache_final"] = {
        "size": torch.backends.cuda.cufft_plan_cache.size,
        "max_size": torch.backends.cuda.cufft_plan_cache.max_size,
    }
    (args.artifact_dir / "final-evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    suite.set("failures", str(failures))
    ET.ElementTree(suite).write(
        args.artifact_dir / "final-evidence.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
