#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then source "$repo_dir/.env"; fi

tilefoundry_repo=${TILEFOUNDRY_REPO:?Set TILEFOUNDRY_REPO in .env}
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
wheel_root=${TILEFOUNDRY_WHEEL_ROOT:-$cache_root/tilefoundry-wheel}
requirements="$repo_dir/config/tilefoundry-runtime-requirements.txt"
requirements_sha256=$(sha256sum "$requirements" | awk '{print $1}')
mkdir -p "$wheel_root"
exec 9>"$wheel_root/.build.lock"
flock 9

if [[ -n ${1:-} || -n ${TILEFOUNDRY_COMMIT:-} ]]; then
    commit=${1:-$TILEFOUNDRY_COMMIT}
else
    remote=upstream
    git -C "$tilefoundry_repo" remote | grep -qx upstream || remote=origin
    git -C "$tilefoundry_repo" fetch --quiet "$remote" main
    commit=$(git -C "$tilefoundry_repo" rev-parse "$remote/main")
fi
commit=$(git -C "$tilefoundry_repo" rev-parse "$commit^{commit}")

uv_bin=${TILEFOUNDRY_UV_BIN:-$(command -v uv || true)}
[[ -n "$uv_bin" ]] || { echo "uv is required to build the admitted wheel" >&2; exit 1; }
builder_root="$wheel_root/builder"
python_bin="$builder_root/bin/python"
if [[ ! -x "$python_bin" ]]; then
    "$uv_bin" venv --seed --no-project --python "${TILEFOUNDRY_BUILD_PYTHON:-3.12}" "$builder_root" >&2
    "$uv_bin" pip install --python "$python_bin" 'setuptools>=68' 'setuptools-scm>=8' >&2
fi

destination="$wheel_root/$commit"
mkdir -p "$destination"
wheel=$(find "$destination" -maxdepth 1 -type f -name 'tilefoundry-*.whl' -print -quit)
if [[ -z "$wheel" ]]; then
    temp_root=$(mktemp -d "${TMPDIR:-/tmp}/tilefoundry-wheel.XXXXXX")
    source_tree="$temp_root/source"
    cleanup() {
        git -C "$tilefoundry_repo" worktree remove --force "$source_tree" >/dev/null 2>&1 || true
        rm -rf -- "$temp_root"
    }
    trap cleanup EXIT
    git -C "$tilefoundry_repo" worktree add --detach "$source_tree" "$commit" >&2
    staging="$temp_root/wheel"
    mkdir -p "$staging"
    "$python_bin" -m pip wheel "$source_tree" --no-deps --no-build-isolation --wheel-dir "$staging" >&2
    wheel=$(find "$staging" -maxdepth 1 -type f -name 'tilefoundry-*.whl' -print -quit)
    [[ -n "$wheel" ]] || { echo "TileFoundry wheel build produced no wheel" >&2; exit 1; }
    cp -- "$wheel" "$destination/"
    wheel="$destination/$(basename -- "$wheel")"
    cleanup
    trap - EXIT
else
    wheel=$(realpath "$wheel")
fi

dependency_root="$wheel_root/deps/$requirements_sha256"
if [[ ! -f "$dependency_root/.complete" ]]; then
    temp_dependencies=$(mktemp -d "${TMPDIR:-/tmp}/tilefoundry-deps.XXXXXX")
    "$python_bin" -m pip download --only-binary=:all: --no-deps \
        --requirement "$requirements" --dest "$temp_dependencies" >&2
    mkdir -p "$dependency_root"
    cp -- "$temp_dependencies"/*.whl "$dependency_root/"
    touch "$dependency_root/.complete"
    rm -rf -- "$temp_dependencies"
fi

wheel=$(realpath "$wheel")
wheel_sha256=$(sha256sum "$wheel" | awk '{print $1}')
tmp_env=$(mktemp "$wheel_root/current.env.XXXXXX")
{
    printf 'export TILEFOUNDRY_WHEEL=%q\n' "$wheel"
    printf 'export TILEFOUNDRY_WHEEL_SHA256=%q\n' "$wheel_sha256"
    printf 'export TILEFOUNDRY_WHEEL_COMMIT=%q\n' "$commit"
    printf 'export TILEFOUNDRY_REQUIREMENTS_SHA256=%q\n' "$requirements_sha256"
} >"$tmp_env"
chmod 0644 "$tmp_env"
mv -- "$tmp_env" "$wheel_root/current.env"
printf '%s\n' "$wheel"
