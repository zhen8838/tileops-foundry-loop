#!/usr/bin/env bash
set -euo pipefail

: "${FOREMAN_WORKTREE:?FOREMAN_WORKTREE is required}"
: "${TILEOPS_FOUNDRY_LOOP_ROOT:?TILEOPS_FOUNDRY_LOOP_ROOT is required}"

git -C "$FOREMAN_WORKTREE" submodule update --init --recursive
"$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/build_tilefoundry_wheel.sh" >/dev/null
# Both of the round's environments are built here, so a worker's pane opens onto
# a `tilefoundry` command and a `tileops-run` container that already exist.
"$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/build_round_venv.sh" "$FOREMAN_WORKTREE" >/dev/null
cd "$FOREMAN_WORKTREE"
"$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/tileops-container.sh" start
