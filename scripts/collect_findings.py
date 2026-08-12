#!/usr/bin/env python3
"""Collect round findings verbatim for later human triage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict] = []
    for path in sorted(args.root.glob("*/findings.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for finding in data.get("findings", []):
            records.append({"round": path.parent.name, **finding})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"findings": records}, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
