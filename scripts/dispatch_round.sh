#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then source "$repo_dir/.env"; fi
(( $# == 3 )) || { echo "usage: $0 TASK BRANCH ABSOLUTE_BRIEF" >&2; exit 2; }
task=$1
branch=$2
brief=$3
[[ "$brief" = /* && -f "$brief" ]] || { echo "brief must be an absolute file" >&2; exit 2; }

if [[ -n ${TILEOPS_PI_BIN:-} ]]; then
    pi_bin=$TILEOPS_PI_BIN
else
    search_path=${PATH#"$repo_dir/scripts:"}
    pi_bin=$(PATH="$search_path" command -v pi || true)
fi
[[ -n "$pi_bin" && -x "$pi_bin" ]] || {
    echo "pi not found; install Pi or set TILEOPS_PI_BIN in .env" >&2
    exit 1
}

"$repo_dir/scripts/build_tilefoundry_wheel.sh" >/dev/null
"$repo_dir/scripts/write_worker_admission.sh" "$task" "$brief"

args=(assign solo --project tileops --prompt "Work in the round described by $brief." \
    --task "$task" --branch "$branch" --kind pi)
[[ -n ${TILEOPS_ROUND_MODEL:-} ]] && args+=(--model "$TILEOPS_ROUND_MODEL")
[[ -n ${TILEOPS_ROUND_EFFORT:-} ]] && args+=(--effort "$TILEOPS_ROUND_EFFORT")
foreman "${args[@]}"
