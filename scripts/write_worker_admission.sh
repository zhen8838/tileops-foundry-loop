#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then source "$repo_dir/.env"; fi

(( $# == 2 )) || { echo "usage: $0 TASK ABSOLUTE_BRIEF" >&2; exit 2; }
task=$1
brief=$2
[[ "$task" =~ ^[a-z][a-z0-9_-]*$ ]] || { echo "invalid task: $task" >&2; exit 2; }
[[ "$brief" = /* && -f "$brief" ]] || { echo "brief must be an absolute file" >&2; exit 2; }

round_host=$(cd -- "$(dirname -- "$brief")" && pwd -P)
round_slug=$(basename -- "$round_host")
loop_state_root=$(cd -- "${TILEOPS_LOOP_STATE_ROOT:-$repo_dir/rounds}" && pwd -P)
[[ "$round_host" == "$loop_state_root"/* ]] || {
    echo "brief must be below TILEOPS_LOOP_STATE_ROOT=$loop_state_root" >&2
    exit 2
}

cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
admission_dir="$cache_root/worker-admissions"
mkdir -p "$admission_dir"
tmp=$(mktemp "$admission_dir/.${task}.XXXXXX")
trap 'rm -f "$tmp"' EXIT
{
    printf 'export TILEOPS_ROUND_HOST=%q\n' "$round_host"
    printf 'export TILEOPS_ROUND_SLUG=%q\n' "$round_slug"
    printf 'export TILEOPS_TASK=%q\n' "$task"
} >"$tmp"
chmod 600 "$tmp"
mv -- "$tmp" "$admission_dir/$task.env"
trap - EXIT
