"""Compare the original CUDA 12.9 evidence with the CUDA 13.2 retest."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


NAMES = ("candidate", "base", "fla")


def geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def trial_median(record: dict, name: str) -> float:
    return statistics.median(
        trial["median_ms"] for trial in record["timings"][name]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--cu132-summary", type=Path, required=True)
    parser.add_argument("--retest", type=Path, action="append", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    old = json.loads(args.old_summary.read_text())
    current = json.loads(args.cu132_summary.read_text())
    retest_records = [
        json.loads(line)
        for path in args.retest
        for line in path.read_text().splitlines()
    ]
    retests = {record["selector"]: record for record in retest_records}
    if len(retests) != len(retest_records):
        raise ValueError("duplicate retest selector")

    old_rows = {row["selector"]: row for row in old["rows"]}
    current_rows = {row["selector"]: row for row in current["rows"]}
    if old_rows.keys() != current_rows.keys() or not retests.keys() <= current_rows.keys():
        raise ValueError("old/current/retest selectors do not match")

    retest_medians = {
        selector: {name: trial_median(record, name) for name in NAMES}
        for selector, record in retests.items()
    }
    adjudicated = {
        name: [
            retest_medians[row_selector][name]
            if row_selector in retest_medians
            else current_rows[row_selector]["timings"][name]["median_ms"]
            for row_selector in sorted(current_rows)
        ]
        for name in NAMES
    }
    adjudicated_geomean = {
        name: geomean(values) for name, values in adjudicated.items()
    }

    rows = []
    for row_selector in sorted(current_rows):
        old_row = old_rows[row_selector]
        new_row = current_rows[row_selector]
        new_medians = {
            name: (
                retest_medians[row_selector][name]
                if row_selector in retest_medians
                else new_row["timings"][name]["median_ms"]
            )
            for name in NAMES
        }
        rows.append(
            {
                "selector": row_selector,
                "adjudicated": row_selector in retest_medians,
                "old_ms": {
                    name: old_row["timings"][name]["median_ms"] for name in NAMES
                },
                "cu132_raw_ms": {
                    name: new_row["timings"][name]["median_ms"] for name in NAMES
                },
                "cu132_reported_ms": new_medians,
                "cu132_over_old": {
                    name: new_medians[name]
                    / old_row["timings"][name]["median_ms"]
                    for name in NAMES
                },
                "cu132_speedup_vs_base": new_medians["base"]
                / new_medians["candidate"],
                "cu132_speedup_vs_fla": new_medians["fla"]
                / new_medians["candidate"],
            }
        )

    result = {
        "row_count": len(rows),
        "adjudications": [
            {
                "selector": selector,
                "reason": "fresh-process five-trial retest of a shared-path bimodal outlier",
                "retest_median_ms": medians,
            }
            for selector, medians in sorted(retest_medians.items())
        ],
        "old_geomean_ms": old["geomean_ms"],
        "cu132_raw_geomean_ms": current["geomean_ms"],
        "cu132_adjudicated_geomean_ms": adjudicated_geomean,
        "cu132_adjudicated_speedup_vs_base": (
            adjudicated_geomean["base"] / adjudicated_geomean["candidate"]
        ),
        "cu132_adjudicated_speedup_vs_fla": (
            adjudicated_geomean["fla"] / adjudicated_geomean["candidate"]
        ),
        "cu132_adjudicated_over_old_geomean": {
            name: adjudicated_geomean[name] / old["geomean_ms"][name]
            for name in NAMES
        },
        "rows": rows,
    }
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "| selector | old C/B/FLA ms | CU132 C/B/FLA ms | CU132/old C/B/FLA | CU132 C/Base | CU132 C/FLA |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        marker = "*" if row["adjudicated"] else ""
        old_ms = row["old_ms"]
        new_ms = row["cu132_reported_ms"]
        ratio = row["cu132_over_old"]
        lines.append(
            f"| {row['selector']}{marker} | "
            f"{old_ms['candidate']:.4f}/{old_ms['base']:.4f}/{old_ms['fla']:.4f} | "
            f"{new_ms['candidate']:.4f}/{new_ms['base']:.4f}/{new_ms['fla']:.4f} | "
            f"{ratio['candidate']:.3f}/{ratio['base']:.3f}/{ratio['fla']:.3f} | "
            f"{row['cu132_speedup_vs_base']:.3f}x | "
            f"{row['cu132_speedup_vs_fla']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "`*` uses the fresh-process five-trial retest; raw main-matrix data remains unchanged.",
            "",
            f"CU132 adjudicated geomean: candidate {adjudicated_geomean['candidate']:.6f} ms; "
            f"base {adjudicated_geomean['base']:.6f} ms; FLA {adjudicated_geomean['fla']:.6f} ms.",
            f"CU132 adjudicated speedup: {result['cu132_adjudicated_speedup_vs_base']:.6f}x vs base; "
            f"{result['cu132_adjudicated_speedup_vs_fla']:.6f}x vs FLA.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
