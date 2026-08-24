#!/usr/bin/env python3
"""Small host-side check; it does not run inside the TileOPs container."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = ("AGENTS.md", "README.md", "brief.md", "knowledge", "evidence", "work")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=Path)
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.round / name).exists()]
    if missing:
        parser.error("round is missing: " + ", ".join(missing))
    print(args.round.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
