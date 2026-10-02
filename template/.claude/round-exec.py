#!/usr/bin/env python3
"""Run a round's shell commands inside the round container.

The agent runs on the host, where its own credentials work and where the round,
the TileOPs worktree and the TileFoundry source already live. Only execution
needs the container: the CUDA stack, the GPU and the installed `tilefoundry`.
Those directories are mounted into the container at the paths they have here, so
a command carries over unchanged -- only its interpreter moves.

Reads one PreToolUse event on stdin and answers on stdout.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

# These act on files and on GitHub, not on the compute stack. They are the same
# program on either side, and the host spares them a round trip.
HOST_COMMANDS = frozenset({"cd", "gh", "git", "ls", "cat", "pwd", "mkdir", "rm", "cp", "mv"})


def ssh_alias(round_dir: Path) -> str | None:
    """The round container's SSH alias, as setup-worktree.sh recorded it."""
    worker_env = round_dir / ".worker-env"
    if not worker_env.is_file():
        return None
    match = re.search(
        r"^export TILEOPS_SSH_ALIAS=(.+)$", worker_env.read_text(), flags=re.MULTILINE
    )
    return shlex.split(match.group(1))[0] if match else None


def runs_on_the_host(command: str) -> bool:
    """Whether every program *command* invokes is a host-side one.

    `cd work && pytest` names a host command first and a container command
    second, so the whole line belongs in the container: one stray segment is
    enough to move it.
    """
    for segment in re.split(r"&&|\|\||[;|\n]", command):
        try:
            words = shlex.split(segment)
        except ValueError:  # unbalanced quotes: let the container decide
            return False
        if words and words[0] not in HOST_COMMANDS:
            return False
    return True


def main() -> int:
    event = json.load(sys.stdin)
    tool_input = event.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return 0

    round_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", event.get("cwd", ".")))
    alias = ssh_alias(round_dir)
    if alias is None or runs_on_the_host(command):
        return 0

    cwd = event.get("cwd") or str(round_dir)
    remote = f"cd {shlex.quote(cwd)} && {command}"
    updated = dict(tool_input)
    updated["command"] = f"ssh -T {shlex.quote(alias)} {shlex.quote(remote)}"
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": updated,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
