#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_dir/.env"
fi

if (( $# != 2 )); then
    echo "usage: $0 TASK ABSOLUTE_BRIEF_PATH" >&2
    exit 2
fi

task=$1
brief=$2
[[ "$task" =~ ^[a-z][a-z0-9_-]*$ ]] || {
    echo "task must use lowercase letters, digits, '-' or '_'" >&2
    exit 2
}
[[ "$brief" = /* && -f "$brief" ]] || {
    echo "brief must be an existing absolute path" >&2
    exit 2
}

round_dir=$(cd -- "$(dirname -- "$brief")" && pwd)
round_slug=$(basename -- "$round_dir")
loop_state_root=$(cd -- "${TILEOPS_LOOP_STATE_ROOT:-$repo_dir/rounds}" && pwd)
[[ "$round_dir" == "$loop_state_root"/* ]] || {
    echo "brief must live below TILEOPS_LOOP_STATE_ROOT: $loop_state_root" >&2
    exit 2
}
[[ "$round_slug" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "round directory has an unsupported name: $round_slug" >&2
    exit 2
}

cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
admission_root="$cache_root/worker-admissions"
mkdir -p "$admission_root"
tmp=$(mktemp "$admission_root/.${task}.XXXXXX")
trap 'rm -f "$tmp"' EXIT
{
    printf 'export TILEOPS_ROUND_HOST=%q\n' "$round_dir"
    printf 'export TILEOPS_ROUND_SLUG=%q\n' "$round_slug"
} >"$tmp"
chmod 600 "$tmp"
mv -f "$tmp" "$admission_root/$task.env"
trap - EXIT
