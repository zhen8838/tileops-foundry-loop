#!/usr/bin/env bash
# Tear a finished round down in the one order that loses nothing: the session
# transcript is copied in first, then container, worktree, pane, admission and
# round venv go -- all of which rebuild in seconds. Safe to re-run.
set -euo pipefail

task=${1:?usage: $0 TASK ROUND_DIR [WORKTREE]}
round=$(cd -- "${2:?usage: $0 TASK ROUND_DIR [WORKTREE]}" && pwd -P)

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_dir/.env"
fi
# shellcheck disable=SC1091
source "$repo_dir/scripts/round-venv.sh"

cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
worktree_root=${TILEOPS_WORKTREE_ROOT:-${TILEOPS_REPO:?TILEOPS_REPO is required}-worktrees}
worktree=${3:-$worktree_root/tileops-$task}

"$repo_dir/scripts/collect_session.sh" "$round"

venv=""
if [[ -d "$worktree" ]]; then
    # The container's name comes from the worktree path, so it goes first.
    ( cd "$worktree" && "$repo_dir/scripts/tileops-container.sh" destroy ) || true
    venv=$(round_venv_path "$worktree")
else
    echo "worktree already gone: $worktree" >&2
fi

foreman done "$task" --rm || echo "foreman has no live job $task" >&2

rm -f "$cache_root/worker-admissions/$task.env"
if [[ -n "$venv" ]]; then
    rm -rf -- "$venv" "$(dirname "$venv")/.$(basename "$venv").lock"
fi

printf 'torn down; session kept in %s\n' "$round/session"
