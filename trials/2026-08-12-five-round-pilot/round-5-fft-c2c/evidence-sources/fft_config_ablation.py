"""CUPTI ablation of existing FFT launch configurations on manifest rows."""

import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel
from tileops.kernels.fft import FFTC2CKernel
from tileops.ops import FFTC2COp


CONFIGS = (
    {"block_size": 128, "threads": 128},
    {"block_size": 256, "threads": 256},
    {"block_size": 512, "threads": 512},
    {"block_size": 1024, "threads": 1024},
    {"block_size": 1024, "threads": 256},
    {"block_size": 1024, "threads": 512},
    {"block_size": 2048, "threads": 512},
    {"block_size": 2048, "threads": 1024},
)
ROWS = (
    ("c64-b1", (4096,), torch.complex64),
    ("c64-b64", (64, 4096), torch.complex64),
    ("c128-b64", (64, 4096), torch.complex128),
)


class Runner:
    def __init__(self, shape, dtype, config):
        self.shape = shape
        self.n = shape[-1]
        self.batch = 1 if len(shape) == 1 else shape[0]
        self.kernel = FFTC2CKernel(
            self.n, self.batch, dtype, config=config, tune=False
        )
        device = torch.device("cuda")
        self.lut_real, self.lut_imag = FFTC2COp._build_lut(
            self.n, dtype, device
        )

    def __call__(self, x):
        pair = torch.view_as_real(x).reshape(self.batch, self.n, 2)
        output = self.kernel.forward_interleaved(pair, self.lut_real, self.lut_imag)
        return torch.view_as_complex(output).reshape(self.shape)


def main():
    root = Path("round5-artifacts")
    records = []
    suite = ET.Element(
        "testsuite", name="fft_config_ablation", tests=str(len(ROWS) * len(CONFIGS))
    )
    failures = 0
    for label, shape, dtype in ROWS:
        torch.manual_seed(31)
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        expected = torch.fft.fft(x)
        tolerance = 1e-4 if dtype is torch.complex64 else 1e-8
        for config in CONFIGS:
            runner = Runner(shape, dtype, config)
            actual = runner(x)
            passed = True
            assertion = None
            try:
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
            except AssertionError as exc:
                passed = False
                assertion = str(exc)
            samples = bench_kernel(
                runner, args=(x,), dry_run_ms=5.0, repeat_ms=30.0
            )
            record = {
                "row": label,
                "shape": list(shape),
                "dtype": str(dtype),
                "config": config,
                "assert_close": passed,
                "assertion": assertion,
                "median_ms": statistics.median(samples),
                "p10_ms": sorted(samples)[round(0.1 * (len(samples) - 1))],
                "p90_ms": sorted(samples)[round(0.9 * (len(samples) - 1))],
                "samples_ms": samples,
                "metadata": _capture_bench_meta(),
            }
            records.append(record)
            case = ET.SubElement(
                suite,
                "testcase",
                name=f"{label}-{config['block_size']}-{config['threads']}",
            )
            if not passed:
                failures += 1
                ET.SubElement(case, "failure", message="assert_close failed").text = assertion
            print(
                label,
                config,
                f"median={record['median_ms']:.6f} ms",
                record["metadata"],
            )
    suite.set("failures", str(failures))
    (root / "config-ablation.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    ET.ElementTree(suite).write(
        root / "config-ablation.xml", encoding="utf-8", xml_declaration=True
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
