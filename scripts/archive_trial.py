#!/usr/bin/env python3
"""Create a reviewable, sanitized archive from a local loop-state directory."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


SKIP_SUFFIXES = {".bundle", ".pyc", ".pt", ".safetensors"}
SKIP_PARTS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}
SKIP_ARCHIVE_PARTS = {
    "base-snapshot",
    "ci-artifacts",
    "ci-gpu-smoke-artifact",
    "cu132-worktree-staging-retired",
    "local-artifacts",
    "worktree-retired",
    "worktree-artifacts",
    "worktree-cleanup-backup",
}
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
HOST_PATH = re.compile(r"/home/[^/\s]+")
TEMP_PATH = re.compile(r"/tmp/[^\s\"']+")
ROOT_PATH = re.compile(r"/root(?=/|\b)")
WORKSPACE_PATH = re.compile(r"/workspace(?=/|\b)")
MOUNT_PATH = re.compile(r"/mnt(?=/|\b)")
EMAIL = re.compile(r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def sanitize(text: str, source: Path) -> str:
    text = text.replace(str(source), "$TRIAL_SOURCE")
    text = HOST_PATH.sub("$HOST_HOME", text)
    text = TEMP_PATH.sub("$TMP_PATH", text)
    text = ROOT_PATH.sub("$ROOT_HOME", text)
    text = WORKSPACE_PATH.sub("$CONTAINER_WORKSPACE", text)
    text = MOUNT_PATH.sub("$MOUNT_ROOT", text)
    text = text.replace("/ci-cache", "$CI_CACHE")
    text = EMAIL.sub("$REDACTED_EMAIL", text)
    had_final_newline = text.endswith(("\n", "\r"))
    text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n")
    return text + ("\n" if had_final_newline else "")


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise SystemExit(f"source is not a directory: {source}")
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    destination.mkdir(parents=True)

    copied: list[tuple[str, int, str]] = []
    skipped: list[tuple[str, str]] = []
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        relative_text = relative.as_posix()
        if source_path.is_dir() or source_path.is_symlink():
            continue
        if any(part in SKIP_PARTS for part in relative.parts):
            skipped.append((relative_text, "cache or repository metadata"))
            continue
        if any(part in SKIP_ARCHIVE_PARTS for part in relative.parts):
            skipped.append((relative_text, "duplicate CI or worktree snapshot"))
            continue
        if any(part == "ci" or part.startswith("ci-") for part in relative.parts):
            skipped.append((relative_text, "duplicate remote CI output"))
            continue
        if source_path.name.startswith("ci-") and source_path.suffix in {".log", ".json"}:
            skipped.append((relative_text, "duplicate remote CI output"))
            continue
        if source_path.suffix.lower() in SKIP_SUFFIXES:
            skipped.append((relative_text, "binary or non-reviewable generated state"))
            continue
        try:
            raw = source_path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append((relative_text, "non-UTF-8 content"))
            continue
        secret = next((pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)), None)
        if secret is not None:
            raise SystemExit(f"credential-like content in {relative_text}: {secret}")
        sanitized = sanitize(text, source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sanitized, encoding="utf-8")
        digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
        copied.append((relative_text, len(sanitized.encode("utf-8")), digest))

    manifest = [
        "# Trial Archive Manifest",
        "",
        "Source paths and personal email addresses were redacted during import.",
        "Binary tensors, caches, Git bundles, and non-UTF-8 files were omitted.",
        "",
        "## Copied",
        "",
        "| Path | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    manifest.extend(f"| `{path}` | {size} | `{digest}` |" for path, size, digest in copied)
    manifest.extend(["", "## Skipped", "", "| Path | Reason |", "| --- | --- |"])
    manifest.extend(f"| `{path}` | {reason} |" for path, reason in skipped)
    (destination / "MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"archived {len(copied)} files; skipped {len(skipped)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
