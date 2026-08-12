#!/usr/bin/env python3
"""Render a TileOPs PR title and body from structured round data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tileops_foundry_loop.pr import PRContractError, load_data, render_pr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    data_path = args.data.resolve()
    output_dir = (args.output_dir or data_path.parent).resolve()
    try:
        rendered = render_pr(load_data(data_path), data_path)
    except PRContractError as error:
        parser.error(str(error))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pr-title.txt").write_text(rendered.title + "\n", encoding="utf-8")
    (output_dir / "pr-body.md").write_text(rendered.body, encoding="utf-8")
    print(output_dir / "pr-title.txt")
    print(output_dir / "pr-body.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
