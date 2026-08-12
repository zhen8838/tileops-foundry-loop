#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if (( $# != 3 )); then
    echo "usage: $0 TASK BRANCH ABSOLUTE_BRIEF_PATH" >&2
    exit 2
fi

task=$1
branch=$2
brief=$3
[[ "$brief" = /* && -f "$brief" ]] || {
    echo "brief must be an existing absolute path" >&2
    exit 2
}

"$repo_dir/scripts/build_tilefoundry_wheel.sh" >/dev/null

foreman assign solo --project tileops \
    --prompt "Execute the closed TileFoundry optimization loop in $brief. Use $repo_dir/PLAYBOOK.md for gates. Do not edit TileFoundry or dispatch another agent." \
    --task "$task" --branch "$branch" \
    --kind codex --model gpt-5.6-sol --effort high
