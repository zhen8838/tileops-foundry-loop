#!/usr/bin/env bash
set -euo pipefail

: "${FOREMAN_WORKTREE:?FOREMAN_WORKTREE is required}"
: "${FOREMAN_TASK:?FOREMAN_TASK is required}"
: "${TILEOPS_FOUNDRY_LOOP_ROOT:?TILEOPS_FOUNDRY_LOOP_ROOT is required}"

source "$TILEOPS_FOUNDRY_LOOP_ROOT/config/defaults.env"
if [[ -f "$TILEOPS_FOUNDRY_LOOP_ROOT/.env" ]]; then
    source "$TILEOPS_FOUNDRY_LOOP_ROOT/.env"
fi
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
admission="$cache_root/worker-admissions/$FOREMAN_TASK.env"
[[ -f "$admission" ]] || { echo "missing worker admission: $admission" >&2; exit 1; }
source "$admission"

"$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/build_tilefoundry_wheel.sh" >/dev/null
cd "$FOREMAN_WORKTREE"
"$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/tileops-container.sh" start
