#!/usr/bin/env python3
"""Drop a declared-but-unimported requirement from an installed distribution.

Runs inside a round container, never on a host checkout. It exists for one
shape of problem: the admitted TileFoundry wheel declares a bound the official
runner image cannot satisfy, on a package no packaged module imports. Honouring
such a bound means replacing part of the admitted baseline stack for something
nothing loads, so the metadata line is removed here instead -- after the wheel
is installed, before the dependency closure is checked, and printed so every
preflight artifact says which line went and from where.

Removing a line for a package the wheel does import would hide a real break, so
each name passed here has to be one the caller has shown is never imported.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import pathlib
import re
import sys


def metadata_path(distribution: metadata.Distribution) -> pathlib.Path:
    for file in distribution.files or ():
        if file.name == "METADATA" and file.parent.name.endswith(".dist-info"):
            return pathlib.Path(str(distribution.locate_file(file))).resolve()
    raise SystemExit("installed distribution has no METADATA file")


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} DISTRIBUTION REQUIREMENT [REQUIREMENT...]", file=sys.stderr)
        return 2
    name, *requirements = sys.argv[1:]

    distribution = metadata.distribution(name)
    target = metadata_path(distribution)
    pattern = re.compile(
        rf"^Requires-Dist:\s*({'|'.join(re.escape(r) for r in requirements)})\b",
        re.IGNORECASE,
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    removed = [line for line in lines if pattern.match(line)]
    if removed:
        kept = [line for line in lines if not pattern.match(line)]
        target.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "distribution": name,
                "version": distribution.version,
                "metadata": str(target),
                "removed": removed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
