"""Refresh structured PR data from the post-rebase performance summary."""

from __future__ import annotations

import json
from pathlib import Path


SUMMARY = Path(".rebase-adjudicated-summary.json")
PR_DATA = Path(
    "$TRIAL_SOURCE/round-3-gated-deltanet-prefill/"
    "pr-data.json"
)


summary = json.loads(SUMMARY.read_text())
data = json.loads(PR_DATA.read_text())
rows = {row["selector"]: row for row in summary["rows"]}
for workload in data["workloads"]:
    dimensions = workload["dimensions"]
    selector = f"{dimensions['S']},{dimensions['H']},{workload['dtype']}"
    row = rows.pop(selector)
    workload["results"] = {
        "candidate": {"median_ms": row["candidate"]},
        "incumbent": {"median_ms": row["incumbent"]},
        "fla": {"median_ms": row["fla"]},
    }
if rows:
    raise ValueError(f"unmatched evidence rows: {sorted(rows)}")

data["method"] = (
    "Candidate, exact incumbent route replay, and FLA ran in one process on "
    "common inputs with the same precision and output/final-state contract, L2 "
    "reset, adaptive repeats, forward/reverse interleaving, and fail-closed "
    "native-CUPTI activity-window attribution; compilation was excluded. Values "
    "are medians of three interleaved trial medians except two shared-path "
    "bimodal rows, which use independent fresh-process medians of five trials."
)
data["limitations"] = [
    "Classification is measured SOTA against contract-equivalent FLA 0.5.2: "
    "the candidate is faster on every primary workload and its 1.892977 ms "
    "geometric mean is lower than FLA's 4.062778 ms.",
    "The candidate geometric mean is 1.0189x faster than the incumbent, but it "
    "is slower on 11 of 26 individual incumbent comparisons; the durable gain "
    "is concentrated in the two 4K rows, while long-context candidate/incumbent "
    "differences mostly measure noise on the same runtime route.",
    "The 131072-token/64-head FP16 and BF16 rows entered shared "
    "candidate/incumbent/FLA bimodal slow states in the full run; their published "
    "values are independent fresh-process five-trial medians, while the original "
    "observations remain part of the internal evidence.",
    "The displayed TileFoundry module is the exact authored one-step semantic "
    "description. A dynamic scan-output insertion lowering gap prevented direct "
    "generation of the shipped prefill kernel, so the delivered optimization "
    "changes selection around the existing TileOps pipeline and is not "
    "represented as generated from this module.",
    "The local-chunk specialization is measured on H200; other supported "
    "architectures retain the existing correctness and fallback surface but may "
    "not share the same tuning optimum.",
]
PR_DATA.write_text(json.dumps(data, indent=2) + "\n")
