#!/usr/bin/env python3
"""Check a round's TileFoundry provenance before rendering a performance PR."""

from __future__ import annotations

import argparse
from pathlib import Path

from tileops_foundry_loop.kernel_diff import check_kernel_diff
from tileops_foundry_loop.round import RoundError, validate_round


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--tileops-repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        provenance = validate_round(args.round_dir)
        check_kernel_diff(
            args.tileops_repo.resolve(),
            args.base,
            args.head,
            provenance["runtime_twin"]["production_kernel"],
        )
    except (RoundError, ValueError) as error:
        parser.error(str(error))
    print("round provenance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
