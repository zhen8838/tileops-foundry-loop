#!/usr/bin/env python3
"""Require a substantive TileLang kernel-body diff before a performance PR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tileops_foundry_loop.kernel_diff import KernelDiffError, check_kernel_diff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    try:
        check_kernel_diff(args.repo.resolve(), args.base, args.head, args.kernel)
    except KernelDiffError as error:
        parser.error(str(error))
    print(f"kernel diff gate passed: {args.kernel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
