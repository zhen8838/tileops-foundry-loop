#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if (( $# < 2 )); then
    echo "usage: $0 PANE REVIEW_TEXT..." >&2
    exit 2
fi

pane=$1
shift
review=$*
foreman say --now "$pane" \
    "Read the Review and Retest section of $repo_dir/PLAYBOOK.md and handle this review in your existing PR/worktree: $review"
