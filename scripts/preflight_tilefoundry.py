#!/usr/bin/env python3
"""Smoke the installed TileFoundry analysis and scheduling surfaces."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


MODEL = '''\
from tilefoundry import func, module
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.target import CudaTarget

@module(
    entry="main",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class AdmissionSmoke:
    @func
    def main(x: Tensor[(1024,), "f32"]):
        return tf.add(x, x)
'''


def run(*arguments: str) -> dict:
    process = subprocess.run(
        ["tilefoundry", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"tilefoundry {' '.join(arguments)} failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    return json.loads(process.stdout)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tilefoundry-admission-") as temporary:
        source = Path(temporary) / "admission_smoke.py"
        source.write_text(MODEL, encoding="utf-8")
        selector = f"{source}:AdmissionSmoke"
        analysis = run(
            "analyze",
            selector,
            "--compute-cost",
            "--memory",
            "--roofline",
            "--timeline",
            "--json",
        )
        schedule = run(
            "schedule",
            selector,
            "--topology",
            "cta",
            "--solver-timeout=15",
            "--solver-workers=1",
            "--first-plan",
            "--json",
        )
    report = {
        "analysis_executed": analysis["executed"],
        "schedule_topology": schedule["topology"],
        "passed": analysis["executed"]
        == ["compute-cost", "memory", "roofline", "timeline"]
        and schedule["topology"] == "cta",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
