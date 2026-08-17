#!/usr/bin/env python3
"""Drop a declared-but-unimported requirement from an installed distribution.

Runs inside a round container, never on a host checkout. Honouring a bound the
official image cannot satisfy, on a package no packaged module imports, means
replacing part of the admitted baseline stack for something nothing loads; the
metadata line goes instead, printed so the preflight artifact says which. Only
pass a name the caller has shown is never imported.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import pathlib
import re
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} DISTRIBUTION REQUIREMENT...", file=sys.stderr)
        return 2
    name, *requirements = sys.argv[1:]

    distribution = metadata.distribution(name)
    target = next(
        (
            pathlib.Path(str(distribution.locate_file(file))).resolve()
            for file in distribution.files or ()
            if file.name == "METADATA" and file.parent.name.endswith(".dist-info")
        ),
        None,
    )
    if target is None:
        raise SystemExit(f"{name} has no METADATA file")

    pattern = re.compile(
        rf"^Requires-Dist:\s*({'|'.join(map(re.escape, requirements))})\b", re.IGNORECASE
    )
    lines = target.read_text(encoding="utf-8").splitlines()
    removed = [line for line in lines if pattern.match(line)]
    if removed:
        target.write_text(
            "\n".join(line for line in lines if not pattern.match(line)) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"distribution": name, "metadata": str(target), "removed": removed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
