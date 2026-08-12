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

foreman assign solo --project tileops \
    --prompt "Set TILEOPS_FOUNDRY_LOOP_ROOT=$repo_dir. Read and follow $repo_dir/PLAYBOOK.md as the round agent, then execute the brief at $brief." \
    --task "$task" --branch "$branch" \
    --kind codex --model gpt-5.6-sol --effort high
