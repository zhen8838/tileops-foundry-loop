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

# Which agent runs the round is a property of the experiment, not of the loop, so
# it is set per dispatch. The defaults are what the recorded rounds so far used.
kind=${TILEOPS_ROUND_KIND:-codex}
model=${TILEOPS_ROUND_MODEL:-gpt-5.6-sol}
effort=${TILEOPS_ROUND_EFFORT:-high}

"$repo_dir/scripts/build_tilefoundry_wheel.sh" >/dev/null
"$repo_dir/scripts/write_worker_admission.sh" "$task" "$brief"

foreman assign solo --project tileops \
    --prompt "Execute the closed TileFoundry optimization loop in $brief. Use $repo_dir/PLAYBOOK.md for gates. Do not edit TileFoundry or dispatch another agent." \
    --task "$task" --branch "$branch" \
    --kind "$kind" --model "$model" --effort "$effort"
