"""Summarize the three-trial Gated DeltaNet CUPTI evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    if len(records) != 26:
        raise ValueError(f"expected 26 records, got {len(records)}")

    rows = []
    for record in records:
        timings = {}
        for name in ("candidate", "base", "fla"):
            trial_medians = [trial["median_ms"] for trial in record["timings"][name]]
            samples = [
                sample
                for trial in record["timings"][name]
                for sample in trial["samples_ms"]
            ]
            timings[name] = {
                "median_ms": statistics.median(trial_medians),
                "trial_median_min_ms": min(trial_medians),
                "trial_median_max_ms": max(trial_medians),
                "trial_relative_range": (
                    max(trial_medians) - min(trial_medians)
                )
                / statistics.median(trial_medians),
                "sample_p10_ms": percentile(samples, 0.1),
                "sample_p90_ms": percentile(samples, 0.9),
                "sample_count": len(samples),
            }
        candidate_ms = timings["candidate"]["median_ms"]
        base_ms = timings["base"]["median_ms"]
        fla_ms = timings["fla"]["median_ms"]
        rows.append(
            {
                "selector": record["selector"],
                "candidate_partition": record["route"]["candidate_partition"],
                "base_partition": record["route"]["base_partition"],
                "timings": timings,
                "speedup_vs_base": base_ms / candidate_ms,
                "speedup_vs_fla": fla_ms / candidate_ms,
                "candidate_max_o_abs": record["errors_vs_fla"]["candidate"][
                    "max_o_abs"
                ],
                "candidate_max_state_abs": record["errors_vs_fla"]["candidate"][
                    "max_state_abs"
                ],
                "base_max_o_abs": record["errors_vs_fla"]["base"]["max_o_abs"],
                "base_max_state_abs": record["errors_vs_fla"]["base"][
                    "max_state_abs"
                ],
                "candidate_peak_bytes": record["errors_vs_fla"]["candidate"][
                    "peak_allocated_delta_bytes"
                ],
                "base_peak_bytes": record["errors_vs_fla"]["base"][
                    "peak_allocated_delta_bytes"
                ],
                "fla_peak_bytes": record["errors_vs_fla"]["fla"][
                    "peak_allocated_delta_bytes"
                ],
            }
        )

    geomeans = {
        name: geometric_mean([row["timings"][name]["median_ms"] for row in rows])
        for name in ("candidate", "base", "fla")
    }
    summary = {
        "row_count": len(rows),
        "rows_candidate_slower_than_fla": [
            row["selector"]
            for row in rows
            if row["timings"]["candidate"]["median_ms"]
            > row["timings"]["fla"]["median_ms"]
        ],
        "geomean_ms": geomeans,
        "geomean_speedup_vs_base": geomeans["base"] / geomeans["candidate"],
        "geomean_speedup_vs_fla": geomeans["fla"] / geomeans["candidate"],
        "max_trial_relative_range": {
            name: max(row["timings"][name]["trial_relative_range"] for row in rows)
            for name in ("candidate", "base", "fla")
        },
        "rows": rows,
    }
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "| S | H | dtype | route C/B | candidate ms [trial range] | base ms [trial range] | FLA ms [trial range] | C/Base | C/FLA | max abs o/state |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        seq_len, heads, dtype = row["selector"].split(",")
        route = (
            f"{'P' if row['candidate_partition'] else 'N'}/"
            f"{'P' if row['base_partition'] else 'N'}"
        )
        cells = []
        for name in ("candidate", "base", "fla"):
            timing = row["timings"][name]
            cells.append(
                f"{timing['median_ms']:.4f} "
                f"[{timing['trial_median_min_ms']:.4f}, "
                f"{timing['trial_median_max_ms']:.4f}]"
            )
        lines.append(
            f"| {seq_len} | {heads} | {dtype} | {route} | {cells[0]} | "
            f"{cells[1]} | {cells[2]} | {row['speedup_vs_base']:.3f}x | "
            f"{row['speedup_vs_fla']:.3f}x | "
            f"{row['candidate_max_o_abs']:.6g}/{row['candidate_max_state_abs']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Geomean latency: candidate {geomeans['candidate']:.6f} ms; "
            f"base {geomeans['base']:.6f} ms; FLA {geomeans['fla']:.6f} ms.",
            f"Geomean speedup: {summary['geomean_speedup_vs_base']:.6f}x vs base; "
            f"{summary['geomean_speedup_vs_fla']:.6f}x vs FLA.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
