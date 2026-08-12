#!/usr/bin/env python3
"""Validate the structured and rendered TileFoundry TileOPs PR contract."""

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
    args = parser.parse_args()
    data_path = args.data.resolve()
    try:
        rendered = render_pr(load_data(data_path), data_path)
    except PRContractError as error:
        print(f"PR contract failed: {error}", file=sys.stderr)
        return 1
    print(f"PR contract OK: {rendered.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
