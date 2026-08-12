#!/usr/bin/env python3
"""Create a round directory from the reusable templates."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = (args.root or repo / "rounds").resolve()
    destination = root / args.slug
    if destination.exists():
        parser.error(f"round already exists: {destination}")
    destination.mkdir(parents=True)
    (destination / "artifacts").mkdir()

    replacements = {
        "{{SLUG}}": args.slug,
        "{{OPERATOR}}": args.operator,
        "{{SCOPE}}": args.scope,
    }
    for source_name, target_name in (
        ("round-brief.md", "brief.md"),
        ("report.md", "report.md"),
    ):
        text = (repo / "templates" / source_name).read_text(encoding="utf-8")
        for before, after in replacements.items():
            text = text.replace(before, after)
        (destination / target_name).write_text(text, encoding="utf-8")
    pr_data = (repo / "templates" / "pr-data.json").read_text(encoding="utf-8")
    for before, after in replacements.items():
        pr_data = pr_data.replace(before, after)
    (destination / "pr-data.json").write_text(pr_data, encoding="utf-8")
    hir_source = (repo / "templates" / "authored_hir.py").read_text(encoding="utf-8")
    (destination / "authored_hir.py").write_text(hir_source, encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
