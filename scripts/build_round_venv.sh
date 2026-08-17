#!/usr/bin/env bash
# Build a round's host environment from the admitted TileFoundry wheel.
#
# Two environments serve one round, and each holds only what it must:
#
#   this venv        the `tilefoundry` command for graph and solver work --
#                    analyze, schedule, spec, models, tutorial -- none of which
#                    touches a GPU or the TileOPs stack;
#   the container    everything that executes the production TileLang path,
#                    including `tilefoundry check` on the runtime twin, because
#                    only the container has TileOPs, TileLang, and the admitted
#                    CUDA stack.
#
# Both install the same admitted wheel, so the command a worker types means the
# same thing on either side, and neither exposes a TileFoundry source tree.
#
# The venv is built outside the worktree: it is environment, not work, and the
# worktree's base-to-head diff is a published artifact.
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
wheel_env="$wheel_root/current.env"
[[ -f "$wheel_env" ]] || {
    echo "no admitted wheel yet; run: $repo_dir/scripts/build_tilefoundry_wheel.sh" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$wheel_env"

venv=$(round_venv_path "$worktree")
mkdir -p "$(dirname "$venv")"
exec 9>"$(dirname "$venv")/.$(basename "$venv").lock"
flock 9

# The stamp names the wheel this venv was built from, so a round that starts
# after a new TileFoundry commit is admitted gets rebuilt rather than silently
# answering with the previous one.
stamp="$venv/.tileops-round-env"
if [[ -f "$stamp" && $(cat "$stamp") == "$TILEFOUNDRY_WHEEL_SHA256" ]]; then
    printf '%s\n' "$venv"
    exit 0
fi

# shellcheck disable=SC1091
source "$repo_dir/scripts/uv-bin.sh"
"$uv_bin" venv --clear --no-project --python "${TILEFOUNDRY_PYTHON:-3.12}" "$venv" >&2
"$uv_bin" pip install --python "$venv/bin/python" "$TILEFOUNDRY_WHEEL" >&2

# Fail here rather than in a worker's first analyze.
"$venv/bin/tilefoundry" --help >/dev/null
"$venv/bin/python" -c 'import pathlib, tilefoundry; \
p = pathlib.Path(tilefoundry.__file__).resolve(); \
assert ".venv" in str(p) or "site-packages" in str(p), p'

printf '%s\n' "$TILEFOUNDRY_WHEEL_SHA256" >"$stamp"
printf '%s\n' "$venv"
