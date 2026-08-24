#!/usr/bin/env bash
set -euo pipefail

round_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tileops_repo=/workspace/tileops

python "$round_dir/check_round.py" \
    --round-dir "$round_dir" \
    --tileops-repo "$tileops_repo" \
    --head HEAD

cat >&2 <<'EOF'

Artifact checks passed. Re-open the actual files before creating the PR:

1. Was final_hir.py actually analyzed and measured, rather than reconstructed after tuning?
2. Do raw analyze and measurement results justify the kept and rejected placements?
3. Does the production diff implement that HIR decision beyond config/tile/launch tuning?

If any answer is no or uncertain, continue the round instead of opening a PR.
EOF
if [[ ${1:-} != --reviewed ]]; then
    echo "PR not opened. Inspect those files, then rerun: ./open_pr.sh --reviewed" >&2
    exit 2
fi
(( $# == 1 )) || {
    echo "usage: ./open_pr.sh [--reviewed]" >&2
    exit 2
}

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
