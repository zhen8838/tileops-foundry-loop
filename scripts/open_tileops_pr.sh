#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# != 4 )); then
    echo "usage: $0 ROUND_DIR TILEOPS_REPO BASE HEAD" >&2
    exit 2
fi

round_dir=$(realpath "$1")
tileops_repo=$(realpath "$2")
base=$3
head=$4

uv run --project "$repo_dir" python "$repo_dir/scripts/check_round.py" "$round_dir" \
    --tileops-repo "$tileops_repo" --base "$base" --head "$head"
uv run --project "$repo_dir" python - "$round_dir/pr-data.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("classification") == "no improvement":
    raise SystemExit("no-improvement rounds cannot open a performance PR")
PY
uv run --project "$repo_dir" python "$repo_dir/scripts/render_pr.py" \
    "$round_dir/pr-data.json" --output-dir "$round_dir"

title=$(<"$round_dir/pr-title.txt")
branch=$(git -C "$tileops_repo" branch --show-current)
[[ -n "$branch" ]] || {
    echo "TileOPs worktree must be on a branch" >&2
    exit 1
}
git -C "$tileops_repo" push --set-upstream origin "$branch"
cd "$tileops_repo"
gh pr create --repo tile-ai/TileOPs --head "$branch" --base main \
    --title "$title" --body-file "$round_dir/pr-body.md"
