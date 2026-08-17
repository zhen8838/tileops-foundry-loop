#!/usr/bin/env bash
# Build a round's host environment from the admitted TileFoundry wheel.
#
# This venv answers `tilefoundry` for the graph and solver work -- analyze,
# schedule, spec, models, tutorial -- and the container answers everything that
# executes the production TileLang path, `check` included, because only it has
# TileOPs, TileLang and the admitted CUDA stack. Same wheel on both sides, no
# TileFoundry source tree on either. Built outside the worktree: environment is
# not work, and the worktree's diff is published.
set -euo pipefail

worktree=${1:?usage: $0 WORKTREE}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_dir/.env"
fi
# shellcheck disable=SC1091
source "$repo_dir/scripts/round-venv.sh"

cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
wheel_root=${TILEFOUNDRY_WHEEL_ROOT:-$cache_root/tilefoundry-wheel}
[[ -f "$wheel_root/current.env" ]] || {
    echo "no admitted wheel yet; run: $repo_dir/scripts/build_tilefoundry_wheel.sh" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$wheel_root/current.env"

venv=$(round_venv_path "$worktree")
mkdir -p "$(dirname "$venv")"
exec 9>"$(dirname "$venv")/.$(basename "$venv").lock"
flock 9

# The stamp names the wheel this venv holds, so a round starting after a new
# commit is admitted rebuilds instead of answering with the previous one.
stamp="$venv/.tileops-round-env"
if [[ -f "$stamp" && $(cat "$stamp") == "$TILEFOUNDRY_WHEEL_SHA256" ]]; then
    printf '%s\n' "$venv"
    exit 0
fi

# shellcheck disable=SC1091
source "$repo_dir/scripts/uv-bin.sh"
"$uv_bin" venv --clear --no-project --python "${TILEFOUNDRY_PYTHON:-3.12}" "$venv" >&2
"$uv_bin" pip install --python "$venv/bin/python" "$TILEFOUNDRY_WHEEL" >&2
"$venv/bin/tilefoundry" --help >/dev/null  # fail here, not in a worker's first analyze

printf '%s\n' "$TILEFOUNDRY_WHEEL_SHA256" >"$stamp"
printf '%s\n' "$venv"
