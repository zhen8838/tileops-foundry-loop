from __future__ import annotations

import json
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

from tileops.manifest import load_manifest
from tileops.ops import FFTC2COp


def timed_ms(fn, *, warmup: int = 5, repeats: int = 20) -> list[float]:
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


def latency(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": quantile(samples, 0.1),
        "p90_ms": quantile(samples, 0.9),
    }


def run_row(row: dict[str, object]) -> dict[str, object]:
    torch.manual_seed(42)
    shape = tuple(row["input_shape"])
    dtype = getattr(torch, row["dtypes"][0])
    x = torch.randn(shape, dtype=dtype, device="cuda")
    op = FFTC2COp(tune=False)

    def incumbent():
        return op(x)

    def official():
        return torch.fft.fft(x, dim=-1)

    plan_cache = torch.backends.cuda.cufft_plan_cache
    plan_cache_before = plan_cache.size
    expected = official()
    plan_cache_after_first = plan_cache.size
    actual = incumbent()
    torch.cuda.synchronize()
    delta = (actual - expected).abs()

    incumbent_samples = timed_ms(incumbent)
    official_samples = timed_ms(official)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        official()
        torch.cuda.synchronize()
    cuda_events = []
    for event in prof.key_averages():
        device_us = float(getattr(event, "self_device_time_total", 0.0))
        if device_us:
            cuda_events.append(
                {
                    "key": event.key,
                    "count": event.count,
                    "self_device_time_us": device_us,
                }
            )
    cuda_events.sort(key=lambda event: event["self_device_time_us"], reverse=True)

    return {
        "label": row["label"],
        "shape": list(shape),
        "dtype": str(dtype),
        "input_stride": list(x.stride()),
        "outputs": {
            "incumbent_shape": list(actual.shape),
            "incumbent_dtype": str(actual.dtype),
            "official_shape": list(expected.shape),
            "official_dtype": str(expected.dtype),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "allclose": bool(
                torch.allclose(
                    actual,
                    expected,
                    atol=1e-4 if dtype == torch.complex64 else 1e-8,
                    rtol=1e-4 if dtype == torch.complex64 else 1e-8,
                )
            ),
        },
        "cufft_plan_cache": {
            "before": plan_cache_before,
            "after_first_official_call": plan_cache_after_first,
            "after_timing": plan_cache.size,
        },
        "latency_ms": {
            "incumbent_event": latency(incumbent_samples),
            "official_event": latency(official_samples),
        },
        "official_profile_cuda_events": cuda_events,
    }


def main() -> None:
    manifest = load_manifest()["FFTC2COp"]
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "manifest_entry": manifest,
                "rows": [run_row(row) for row in manifest["workloads"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
