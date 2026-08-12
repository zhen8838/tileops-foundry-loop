"""Post-rebase FFT comparison using the current benchmark timing contract."""

import argparse
import json
import linecache
import math
import statistics
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from benchmarks.benchmark_base import ManifestBenchmark
from tileops.ops import FFTC2COp
from workloads.fft import FFTWorkload


ROWS = (
    ("fft-4k-c64-unbatched", (4096,), torch.complex64),
    ("fft-4k-c64-b64", (64, 4096), torch.complex64),
    ("fft-4k-c128-b64", (64, 4096), torch.complex128),
)


def load_base_kernel(revision: str):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/tileops/kernels/fft.py"], text=True
    )
    source = source.replace(
        '"top::fft_c2c_wrapped_kernel"', '"top::fft_c2c_base_wrapped_kernel"'
    ).replace("_fft_c2c_wrapped_kernel", "_fft_c2c_base_wrapped_kernel")
    name = "tileops.kernels.fft_base_snapshot_post_rebase"
    module = types.ModuleType(name)
    module.__file__ = f"{revision}:src/tileops/kernels/fft.py"
    module.__package__ = "tileops.kernels"
    sys.modules[name] = module
    linecache.cache[module.__file__] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        module.__file__,
    )
    exec(compile(source, module.__file__, "exec"), module.__dict__)
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


def correctness(actual, expected, tolerance):
    delta = (actual - expected).abs()
    try:
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    except AssertionError as exc:
        return {
            "passed": False,
            "assertion": str(exc),
            "max_complex_abs": float(delta.max()),
            "mean_complex_abs": float(delta.mean()),
        }
    return {
        "passed": True,
        "assertion": None,
        "max_complex_abs": float(delta.max()),
        "mean_complex_abs": float(delta.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_module = load_base_kernel(args.base_revision)
    payload = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "method": (
            "ManifestBenchmark.compare with native CUPTI window attribution, "
            "forward/reverse drift balancing within each comparison, and rotated "
            "three-way insertion order across trials"
        ),
        "trials": args.trials,
        "rows": {},
    }
    suite = ET.Element("testsuite", name="fft_post_rebase_comparison")
    failures = 0
    test_count = 0

    for row_index, (label, shape, dtype) in enumerate(ROWS):
        torch.manual_seed(101 + row_index)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        candidate = FFTC2COp(tune=False)
        incumbent = BaseRunner(base_module, shape, dtype)
        cufft = lambda value: torch.fft.fft(value, dim=-1)
        implementations = {
            "candidate": candidate,
            "incumbent": incumbent,
            "cufft": cufft,
        }

        candidate(x)
        incumbent(x)
        cufft(x)
        torch.cuda.synchronize()
        expected = cufft(x)
        tolerance = 1e-4 if dtype is torch.complex64 else 1e-8
        row = {
            "shape": list(shape),
            "dtype": str(dtype),
            "candidate_config": candidate.kernel.config,
            "correctness": {},
            "trials": [],
        }
        for name, fn in implementations.items():
            result = correctness(fn(x), expected, tolerance)
            row["correctness"][name] = result
            test_count += 1
            case = ET.SubElement(suite, "testcase", name=f"{label}-{name}")
            if not result["passed"]:
                failures += 1
                ET.SubElement(case, "failure", message="assert_close failed").text = result[
                    "assertion"
                ]

        workload = FFTWorkload(shape[-1], dtype, batch_shape=shape[:-1])
        benchmark = ManifestBenchmark("FFTC2COp", candidate, workload)
        names = list(implementations)
        for trial in range(args.trials):
            offset = trial % len(names)
            insertion_order = names[offset:] + names[:offset]
            ordered = {name: implementations[name] for name in insertion_order}
            results = benchmark.compare(ordered, x)
            for name, result in results.items():
                test_count += 1
                case = ET.SubElement(
                    suite, "testcase", name=f"{label}-{name}-timing-{trial}"
                )
                if result.get("timing") != "cupti":
                    failures += 1
                    ET.SubElement(case, "failure", message="timing is not CUPTI")
            row["trials"].append(
                {
                    "trial": trial,
                    "insertion_order": insertion_order,
                    "results": results,
                }
            )

        row["reported_median_ms"] = {
            name: statistics.median(
                trial["results"][name]["latency_ms"] for trial in row["trials"]
            )
            for name in names
        }
        payload["rows"][label] = row
        print(label, row["reported_median_ms"])

    suite.set("tests", str(test_count))
    suite.set("failures", str(failures))
    suite.set("errors", "0")
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    ET.ElementTree(suite).write(
        args.output_dir / "comparison.xml", encoding="utf-8", xml_declaration=True
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
