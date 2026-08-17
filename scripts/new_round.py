#!/usr/bin/env python3
"""Create a round directory holding its brief, and nothing else.

A round used to start from filled-in copies of an authored HIR, a runtime twin,
a report and three JSON records. Every one of them was a graph or a shape to
copy, which is the opposite of what the loop asks for -- the worker authors the
description, and `scripts/check_round.py` states the contract by refusing what is
missing. So the scaffold is gone: what a round gets is a brief and the commands.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tileops-repo", type=Path, required=True)
    parser.add_argument("--tilefoundry-repo", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = (args.root or repo / "rounds").resolve()
    destination = root / args.slug
    if destination.exists():
        parser.error(f"round already exists: {destination}")
    destination.mkdir(parents=True)

    replacements = {
        "{{SLUG}}": args.slug,
        "{{OPERATOR}}": args.operator,
        "{{SCOPE}}": args.scope,
        "{{BASELINE}}": args.baseline,
        "{{TILEOPS_BASE}}": git_commit(args.tileops_repo.resolve()),
        "{{TILEFOUNDRY_COMMIT}}": git_commit(args.tilefoundry_repo.resolve()),
        "{{ROUND_DIR}}": str(destination),
    }
    brief = (repo / "templates" / "round-brief.md").read_text(encoding="utf-8")
    for before, after in replacements.items():
        brief = brief.replace(before, after)
    (destination / "brief.md").write_text(brief, encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
