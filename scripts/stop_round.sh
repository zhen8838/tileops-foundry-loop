#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
(( $# == 3 )) || { echo "usage: $0 TASK ROUND_DIR WORKTREE" >&2; exit 2; }
task=$1
round_dir=$2
worktree=$3
if [[ -d "$worktree" ]]; then
    (
        cd "$worktree"
        export FOREMAN_TASK=$task
        "$repo_dir/scripts/tileops-container.sh" destroy
    ) || true
fi
foreman done "$task" || true
printf 'container stopped; kept round=%s worktree=%s\n' "$round_dir" "$worktree"
