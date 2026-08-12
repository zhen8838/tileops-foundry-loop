"""Blind-phase radix-2 TileLang runtime twin and evidence runner for FFTC2COp."""

import argparse
import functools
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import tilelang
import tilelang.language as T
import torch

from benchmarks.benchmark_base import _capture_bench_meta, bench_kernel


@functools.lru_cache(maxsize=16)
def build_fft_runtime(n: int, batch: int, scalar_dtype: str, threads: int):
    """Build an in-place radix-2 DIT graph with an explicit bit-reversed load."""
    log2n = int(math.log2(n))

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
        compile_flags=["-O3"],
    )
    def _kernel():
        @T.prim_func
        def fft_radix2(
            x_real: T.Tensor([batch, n], scalar_dtype),
            x_imag: T.Tensor([batch, n], scalar_dtype),
            twiddle_real: T.Tensor([n - 1], scalar_dtype),
            twiddle_imag: T.Tensor([n - 1], scalar_dtype),
            bit_reverse: T.Tensor([n], "int32"),
            output: T.Tensor([batch, n, 2], scalar_dtype),
        ):
            with T.Kernel(batch, threads=threads) as bid:
                shared_r = T.alloc_shared([n], scalar_dtype)
                shared_i = T.alloc_shared([n], scalar_dtype)

                for dst in T.Parallel(n):
                    src = bit_reverse[dst]
                    shared_r[dst] = x_real[bid, src]
                    shared_i[dst] = x_imag[bid, src]
                T.sync_threads()

                for stage in range(log2n):
                    half = 1 << stage
                    width = half * 2
                    offset = half - 1
                    for pair in T.Parallel(n // 2):
                        group = pair // half
                        j = pair % half
                        even = group * width + j
                        odd = even + half
                        even_r = shared_r[even]
                        even_i = shared_i[even]
                        odd_r = shared_r[odd]
                        odd_i = shared_i[odd]
                        tw_r = twiddle_real[offset + j]
                        tw_i = twiddle_imag[offset + j]
                        prod_r = odd_r * tw_r - odd_i * tw_i
                        prod_i = odd_r * tw_i + odd_i * tw_r
                        shared_r[even] = even_r + prod_r
                        shared_i[even] = even_i + prod_i
                        shared_r[odd] = even_r - prod_r
                        shared_i[odd] = even_i - prod_i
                    T.sync_threads()

                for k in T.Parallel(n):
                    output[bid, k, 0] = shared_r[k]
                    output[bid, k, 1] = shared_i[k]

        return fft_radix2

    return _kernel()


def _bit_reverse(n: int, device: torch.device) -> torch.Tensor:
    bits = int(math.log2(n))
    values = []
    for index in range(n):
        value = index
        result = 0
        for _ in range(bits):
            result = (result << 1) | (value & 1)
            value >>= 1
        values.append(result)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _twiddles(n: int, real_dtype: torch.dtype, device: torch.device):
    angles = torch.zeros(n - 1, dtype=torch.float64)
    for stage in range(int(math.log2(n))):
        half = 1 << stage
        k = torch.arange(half, dtype=torch.float64)
        angles[half - 1 : 2 * half - 1] = -2.0 * math.pi * k / (2 * half)
    return torch.cos(angles).to(real_dtype).to(device), torch.sin(angles).to(
        real_dtype
    ).to(device)


class BlindFFT:
    """Public-call-shaped twin including real/imag materialization."""

    def __init__(self, n: int, batch: int, dtype: torch.dtype, threads: int = 256):
        self.n = n
        self.batch = batch
        self.dtype = dtype
        scalar_dtype = "float32" if dtype is torch.complex64 else "float64"
        self.kernel = build_fft_runtime(n, batch, scalar_dtype, threads)
        real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
        device = torch.device("cuda")
        self.twiddle_real, self.twiddle_imag = _twiddles(n, real_dtype, device)
        self.bit_reverse = _bit_reverse(n, device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real.contiguous().reshape(self.batch, self.n)
        x_imag = x.imag.contiguous().reshape(self.batch, self.n)
        out = self.kernel(
            x_real,
            x_imag,
            self.twiddle_real,
            self.twiddle_imag,
            self.bit_reverse,
        )
        return torch.view_as_complex(out).reshape(x.shape)


def _direct_dft(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    real_dtype = x.real.dtype
    t = torch.arange(n, dtype=real_dtype, device=x.device)[:, None]
    k = torch.arange(n, dtype=real_dtype, device=x.device)[None, :]
    theta = -2.0 * math.pi * t * k / n
    w_r, w_i = torch.cos(theta), torch.sin(theta)
    xr, xi = x.real.reshape(-1, n), x.imag.reshape(-1, n)
    yr = xr @ w_r - xi @ w_i
    yi = xr @ w_i + xi @ w_r
    return torch.complex(yr, yi).reshape(x.shape)


def _comparison(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict:
    delta = (actual - expected).abs()
    passed = True
    assertion = None
    try:
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
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


def _quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-trials", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--junit-out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(23)
    small_x = torch.randn(2, 8, dtype=torch.complex64, device="cuda")
    small_op = BlindFFT(8, 2, torch.complex64, threads=32)
    small_actual = small_op(small_x)
    small_expected = _direct_dft(small_x)
    small = _comparison(small_actual, small_expected, 1e-4)

    production_x = torch.randn(4096, dtype=torch.complex64, device="cuda")
    production_op = BlindFFT(4096, 1, torch.complex64, threads=256)
    production_actual = production_op(production_x)
    production_expected = torch.fft.fft(production_x, dim=-1)
    production = _comparison(production_actual, production_expected, 1e-4)

    source = production_op.kernel.get_kernel_source()
    args.source_out.write_text(source, encoding="utf-8")

    trials = []
    for _ in range(args.bench_trials):
        samples = bench_kernel(
            production_op,
            args=(production_x,),
            dry_run_ms=10.0,
            repeat_ms=100.0,
        )
        trials.append(
            {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p10_ms": _quantile(samples, 0.1),
                "p90_ms": _quantile(samples, 0.9),
                "metadata": _capture_bench_meta(),
            }
        )

    payload = {
        "candidate": {
            "algorithm": "radix2_dit_bit_reversed_load",
            "threads": 256,
            "runtime_value_cache": False,
            "shape_only_cache": ["twiddle_real", "twiddle_imag", "bit_reverse"],
            "generated_source": str(args.source_out),
        },
        "small_multistage_direct_dft": small,
        "manifest_4096_b1_complex64": {**production, "latency_trials": trials},
    }
    rendered = json.dumps(payload, indent=2)
    args.json_out.write_text(rendered + "\n", encoding="utf-8")

    suite = ET.Element("testsuite", name="fft_blind_candidate", tests="2")
    failures = 0
    for name, record in (
        ("small_multistage_direct_dft", small),
        ("manifest_4096_b1_complex64", production),
    ):
        case = ET.SubElement(suite, "testcase", name=name)
        if not record["assert_close"]:
            failures += 1
            ET.SubElement(case, "failure", message="assert_close failed").text = record[
                "assertion"
            ]
    suite.set("failures", str(failures))
    ET.ElementTree(suite).write(
        args.junit_out, encoding="utf-8", xml_declaration=True
    )
    print(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
