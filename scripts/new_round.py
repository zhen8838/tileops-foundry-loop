#!/usr/bin/env python3
"""Create a round directory from the reusable templates."""

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
    (destination / "artifacts").mkdir()

    replacements = {
        "{{SLUG}}": args.slug,
        "{{OPERATOR}}": args.operator,
        "{{SCOPE}}": args.scope,
        "{{BASELINE}}": args.baseline,
        "{{TILEOPS_BASE}}": git_commit(args.tileops_repo.resolve()),
        "{{TILEFOUNDRY_COMMIT}}": git_commit(args.tilefoundry_repo.resolve()),
        "{{ROUND_DIR}}": str(destination),
    }
    for source_name, target_name in (
        ("round-brief.md", "brief.md"),
        ("report.md", "report.md"),
        ("authored_hir.py", "authored_hir.py"),
        ("runtime_twin.py", "runtime_twin.py"),
        ("provenance.json", "provenance.json"),
        ("findings.json", "findings.json"),
    ):
        text = (repo / "templates" / source_name).read_text(encoding="utf-8")
        for before, after in replacements.items():
            text = text.replace(before, after)
        (destination / target_name).write_text(text, encoding="utf-8")
    pr_data = (repo / "templates" / "pr-data.json").read_text(encoding="utf-8")
    for before, after in replacements.items():
        pr_data = pr_data.replace(before, after)
    (destination / "pr-data.json").write_text(pr_data, encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
