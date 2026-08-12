"""PyTorch profiler traces for final FFT candidate, base, and cuFFT."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from fft_final_evidence import BaseRunner, ROWS, compare, load_base_kernel
from tileops.ops import FFTC2COp


def profile_one(fn, x, trace_path):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        output = fn(x)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    events = []
    for event in prof.key_averages():
        events.append(
            {
                "key": event.key,
                "count": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "device_time_total_us": getattr(event, "device_time_total", 0.0),
                "cpu_memory_usage": event.cpu_memory_usage,
                "device_memory_usage": event.device_memory_usage,
            }
        )
    return output, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    base_module = load_base_kernel(args.base_source)

    payload = {"rows": {}}
    suite = ET.Element("testsuite", name="fft_profiler_evidence", tests="9")
    failures = 0
    for index, (label, shape, dtype) in enumerate(ROWS):
        torch.manual_seed(101 + index)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        candidate = FFTC2COp(tune=False)
        base = BaseRunner(base_module, shape, dtype)
        reference = lambda value: torch.fft.fft(value, dim=-1)
        implementations = {
            "candidate": candidate,
            "base-incumbent": base,
            "torch-cufft": reference,
        }
        for fn in implementations.values():
            fn(x)
        torch.cuda.synchronize()
        expected = reference(x)
        tolerance = 1e-4 if dtype is torch.complex64 else 1e-8
        row = {}
        for name, fn in implementations.items():
            trace_path = args.artifact_dir / f"{label}-{name}-trace.json"
            output, events = profile_one(fn, x, trace_path)
            correctness = compare(output, expected, tolerance)
            row[name] = {
                "trace": str(trace_path),
                "events": events,
                "correctness": correctness,
            }
            case = ET.SubElement(suite, "testcase", name=f"{label}-{name}")
            if not correctness["assert_close"]:
                failures += 1
                ET.SubElement(case, "failure", message="assert_close failed").text = correctness[
                    "assertion"
                ]
        payload["rows"][label] = row
        print(label, {name: len(record["events"]) for name, record in row.items()})

    (args.artifact_dir / "profiler-evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    suite.set("failures", str(failures))
    ET.ElementTree(suite).write(
        args.artifact_dir / "profiler-evidence.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
