#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_dir/.env"
fi

tilefoundry_repo=${TILEFOUNDRY_REPO:?Set TILEFOUNDRY_REPO in .env}
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
wheel_root=${TILEFOUNDRY_WHEEL_ROOT:-$cache_root/tilefoundry-wheel}
requirements="$repo_dir/config/tilefoundry-runtime-requirements.txt"
requirements_sha256=$(sha256sum "$requirements" | awk '{print $1}')
dependency_root="$wheel_root/deps/$requirements_sha256"
mkdir -p "$wheel_root"
exec 9>"$wheel_root/.build.lock"
flock 9
commit=${1:-${TILEFOUNDRY_COMMIT:-$(git -C "$tilefoundry_repo" rev-parse HEAD)}}
commit=$(git -C "$tilefoundry_repo" rev-parse "$commit^{commit}")
# The build interpreter belongs to the loop, not to the TileFoundry checkout.
# That checkout's `.venv` is rebuilt from a pinned resolution by foreman's
# prepare-worktree hook, which creates it with `uv venv` and therefore without
# pip; borrowing it also made a wheel build depend on how another project
# happens to arrange its environment. This one is seeded once under the wheel
# cache and holds only the build backend TileFoundry declares.
builder_root=$wheel_root/builder
python_bin="$builder_root/bin/python"
if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$repo_dir/scripts/uv-bin.sh"
    rm -rf -- "$builder_root"
    "$uv_bin" venv --seed --no-project \
        --python "${TILEFOUNDRY_BUILD_PYTHON:-3.12}" "$builder_root" >&2
    # `pip wheel` below runs without build isolation, so the backend named in
    # TileFoundry's `[build-system]` has to be present here.
    "$uv_bin" pip install --python "$python_bin" \
        'setuptools>=68' 'setuptools-scm>=8' >&2
fi

destination="$wheel_root/$commit"
current_env="$wheel_root/current.env"
mkdir -p "$destination"
existing=("$destination"/tilefoundry-*.whl)
if (( ${#existing[@]} == 1 )) && [[ -f ${existing[0]} ]]; then
    wheel=${existing[0]}
else
    temp_root=$(mktemp -d "${TMPDIR:-/tmp}/tilefoundry-wheel.XXXXXX")
    source_tree="$temp_root/source"
    staging="$temp_root/wheel"
    source_added=0
    cleanup() {
        if (( source_added )); then
            git -C "$tilefoundry_repo" worktree remove --force "$source_tree" >/dev/null 2>&1 || true
        fi
        rm -rf -- "$temp_root"
    }
    trap cleanup EXIT
    git -C "$tilefoundry_repo" worktree add --detach "$source_tree" "$commit" >&2
    source_added=1
    mkdir -p "$staging"
    "$python_bin" -m pip wheel "$source_tree" --no-deps --no-build-isolation \
        --wheel-dir "$staging" >&2
    built=("$staging"/tilefoundry-*.whl)
    (( ${#built[@]} == 1 )) && [[ -f ${built[0]} ]] || {
        echo "Expected one TileFoundry wheel, found ${#built[@]}" >&2
        exit 1
    }
    cp -- "${built[0]}" "$destination/"
    wheel="$destination/$(basename -- "${built[0]}")"
    cleanup
    trap - EXIT
fi

if [[ ! -f "$dependency_root/.complete" ]]; then
    temp_dependencies=$(mktemp -d "${TMPDIR:-/tmp}/tilefoundry-deps.XXXXXX")
    cleanup_dependencies() {
        rm -rf -- "$temp_dependencies"
    }
    trap cleanup_dependencies EXIT
    "$python_bin" -m pip download --only-binary=:all: --no-deps \
        --requirement "$requirements" --dest "$temp_dependencies" >&2
    mkdir -p "$dependency_root"
    cp -- "$temp_dependencies"/*.whl "$dependency_root/"
    touch "$dependency_root/.complete"
    cleanup_dependencies
    trap - EXIT
fi

wheel=$(realpath "$wheel")
wheel_sha256=$(sha256sum "$wheel" | awk '{print $1}')
temp_env=$(mktemp "$wheel_root/current.env.XXXXXX")
{
    printf 'export TILEFOUNDRY_WHEEL=%q\n' "$wheel"
    printf 'export TILEFOUNDRY_WHEEL_SHA256=%q\n' "$wheel_sha256"
    printf 'export TILEFOUNDRY_WHEEL_COMMIT=%q\n' "$commit"
    printf 'export TILEFOUNDRY_REQUIREMENTS_SHA256=%q\n' "$requirements_sha256"
} >"$temp_env"
chmod 0644 "$temp_env"
mv -- "$temp_env" "$current_env"
printf '%s\n' "$wheel"
