#!/usr/bin/env python3
"""Verify the complete active dependency closure of the installed TileFoundry."""

from __future__ import annotations

import importlib.metadata as metadata
import json
from collections import deque

from packaging.requirements import Requirement
from packaging.version import Version


def main() -> int:
    pending = deque(["tilefoundry"])
    visited: set[str] = set()
    errors: list[dict[str, str]] = []
    versions: dict[str, str] = {}

    while pending:
        requested = pending.popleft()
        normalized = requested.lower().replace("_", "-")
        if normalized in visited:
            continue
        visited.add(normalized)
        try:
            distribution = metadata.distribution(requested)
        except metadata.PackageNotFoundError:
            errors.append({"package": requested, "error": "missing"})
            continue
        name = distribution.metadata["Name"]
        versions[name] = distribution.version
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                errors.append(
                    {
                        "package": name,
                        "requirement": str(requirement),
                        "error": "dependency is missing",
                    }
                )
                continue
            if requirement.specifier and Version(installed) not in requirement.specifier:
                errors.append(
                    {
                        "package": name,
                        "requirement": str(requirement),
                        "installed": installed,
                        "error": "version does not satisfy requirement",
                    }
                )
                continue
            pending.append(requirement.name)

    report = {
        "root": "tilefoundry",
        "checked_packages": dict(sorted(versions.items())),
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
