#!/usr/bin/env python3
"""Create a round by copying the versioned round template."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def replace_tokens(path: Path, values: dict[str, str]) -> None:
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    for token, value in values.items():
        text = text.replace("{{" + token + "}}", value)
    path.write_text(text, encoding="utf-8")


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
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", args.slug):
        parser.error(
            "slug must start with a letter and contain only lowercase letters, "
            "digits, '-' or '_'"
        )

    loop_root = Path(__file__).resolve().parents[1]
    configured_root = os.environ.get("TILEOPS_LOOP_STATE_ROOT")
    root = (args.root or configured_root or loop_root / "rounds")
    root = Path(root).expanduser().resolve()
    destination = root / args.slug
    if destination.exists():
        parser.error(f"round already exists: {destination}")

    values = {
        "SLUG": args.slug,
        "OPERATOR": args.operator,
        "SCOPE": args.scope,
        "BASELINE": args.baseline,
        "TILEOPS_BASE": git_commit(args.tileops_repo.resolve()),
        "TILEFOUNDRY_COMMIT": git_commit(args.tilefoundry_repo.resolve()),
    }

    template = loop_root / "templates" / "round"
    if not template.is_dir():
        parser.error(f"missing round template: {template}")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination)
    for path in destination.rglob("*"):
        replace_tokens(path, values)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
