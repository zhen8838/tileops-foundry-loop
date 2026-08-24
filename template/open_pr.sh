#!/usr/bin/env bash
set -euo pipefail

round_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tileops_repo=/workspace/tileops

python "$round_dir/check_round.py" \
    --round-dir "$round_dir" \
    --tileops-repo "$tileops_repo" \
    --head HEAD

branch=$(git -C "$tileops_repo" branch --show-current)
[[ -n "$branch" ]] || {
    echo "TileOPs worktree is not on a branch" >&2
    exit 1
}
title=$(<"$round_dir/pr-title.txt")

git -C "$tileops_repo" push -u origin "$branch"
if gh pr view "$branch" --repo tile-ai/TileOPs >/dev/null 2>&1; then
    gh pr edit "$branch" --repo tile-ai/TileOPs \
        --title "$title" --body-file "$round_dir/pr-body.md"
else
    gh pr create --repo tile-ai/TileOPs --base main \
        --head "zhen8838:$branch" --title "$title" --body-file "$round_dir/pr-body.md"
fi
gh pr checks "$branch" --repo tile-ai/TileOPs --watch --interval 300
