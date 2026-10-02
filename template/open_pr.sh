#!/usr/bin/env bash
set -euo pipefail

round_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# setup-worktree.sh records the worktree this round builds against.
# shellcheck disable=SC1091
source "$round_dir/.worker-env"
tileops_repo=${TILEOPS_TILEOPS_REPO:?.worker-env does not name the TileOPs worktree}

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
pr_number=
pr_state=
if pr_info=$(gh pr view "$branch" --repo tile-ai/TileOPs \
    --json number,state --jq '[.number, .state] | @tsv' 2>/dev/null); then
    IFS=$'\t' read -r pr_number pr_state <<<"$pr_info"
fi
case "$pr_state" in
    OPEN)
        gh pr edit "$pr_number" --repo tile-ai/TileOPs \
            --title "$title" --body-file "$round_dir/pr-body.md"
        ;;
    CLOSED)
        gh pr reopen "$pr_number" --repo tile-ai/TileOPs
        gh pr edit "$pr_number" --repo tile-ai/TileOPs \
            --title "$title" --body-file "$round_dir/pr-body.md"
        ;;
    MERGED|"")
        gh pr create --repo tile-ai/TileOPs --base main \
            --head "zhen8838:$branch" --title "$title" \
            --body-file "$round_dir/pr-body.md"
        ;;
    *)
        echo "unsupported PR state for #$pr_number: $pr_state" >&2
        exit 1
        ;;
esac
gh pr checks "$branch" --repo tile-ai/TileOPs --watch --interval 300
