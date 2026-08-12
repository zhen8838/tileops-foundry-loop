#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_dir/.env"
fi

image=${TILEOPS_RUNNER_IMAGE:?TILEOPS_RUNNER_IMAGE is required}
env_schema=${TILEOPS_ENV_SCHEMA:?TILEOPS_ENV_SCHEMA is required}
default_tileops=${TILEOPS_REPO:-}
tilefoundry_root=${TILEFOUNDRY_REPO:-}
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}

if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then
    "$TILEOPS_DOCKER_BOOTSTRAP"
fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then
    export DOCKER_HOST=$TILEOPS_DOCKER_HOST
fi

worktree=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ ! -f "$worktree/src/tileops/__init__.py" ]]; then
    worktree=$default_tileops
fi
if [[ -z "$worktree" || ! -f "$worktree/pyproject.toml" ]]; then
    echo "Run from a TileOPs worktree or set TILEOPS_REPO" >&2
    exit 1
fi
if [[ -z "$tilefoundry_root" || ! -f "$tilefoundry_root/pyproject.toml" ]]; then
    echo "Set TILEFOUNDRY_REPO to a TileFoundry checkout" >&2
    exit 1
fi

git_common_dir=$(git -C "$worktree" rev-parse --path-format=absolute --git-common-dir)
mkdir -p "$cache_root"
container_key=$(printf '%s' "$worktree" | sha256sum)
container_name="tileops-foundry-loop-${container_key:0:12}"

destroy_container() {
    if ! docker container inspect "$container_name" >/dev/null 2>&1; then
        return
    fi
    mounted_root=$(docker container inspect \
        --format '{{index .Config.Labels "tileops-foundry-loop.worktree"}}' "$container_name")
    if [[ "$mounted_root" != "$worktree" ]]; then
        echo "Refusing to remove $container_name: label points to $mounted_root" >&2
        exit 1
    fi
    docker rm -f "$container_name" >/dev/null
}

ensure_container() {
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "Runner image is not local; run scripts/preflight.sh first: $image" >&2
        exit 1
    fi

    requested_gpu=${TILEOPS_GPU:-}
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        mounted_root=$(docker container inspect \
            --format '{{index .Config.Labels "tileops-foundry-loop.worktree"}}' "$container_name")
        gpu=$(docker container inspect \
            --format '{{index .Config.Labels "tileops-foundry-loop.gpu"}}' "$container_name")
        actual_image=$(docker container inspect --format '{{.Image}}' "$container_name")
        expected_image=$(docker image inspect --format '{{.Id}}' "$image")
        actual_schema=$(docker container inspect \
            --format '{{index .Config.Labels "tileops-foundry-loop.env-schema"}}' "$container_name")
        [[ "$mounted_root" == "$worktree" ]] || {
            echo "$container_name belongs to $mounted_root" >&2
            exit 1
        }
        [[ "$actual_image" == "$expected_image" && "$actual_schema" == "$env_schema" ]] || {
            echo "$container_name is stale; run: $0 recreate" >&2
            exit 1
        }
        [[ -z "$requested_gpu" || "$requested_gpu" == "$gpu" ]] || {
            echo "$container_name is pinned to GPU $gpu, not $requested_gpu" >&2
            exit 1
        }
        if [[ $(docker container inspect --format '{{.State.Running}}' "$container_name") != true ]]; then
            docker start "$container_name" >/dev/null
        fi
        return
    fi

    gpu=$requested_gpu
    if [[ -z "$gpu" ]]; then
        gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | tr -d ' ' | tr ',' ' ' | sort -k2,2n -k1,1n | head -n 1 | awk '{print $1}')
    fi
    [[ "$gpu" =~ ^[0-9]+$ ]] || {
        echo "Set TILEOPS_GPU to a numeric GPU index" >&2
        exit 1
    }

    docker run --detach --init --name "$container_name" \
        --label "tileops-foundry-loop.worktree=$worktree" \
        --label "tileops-foundry-loop.gpu=$gpu" \
        --label "tileops-foundry-loop.env-schema=$env_schema" \
        --device "nvidia.com/gpu=$gpu" --ipc=host --shm-size=16g \
        --volume "$worktree:/workspace/tileops" \
        --volume "$git_common_dir:$git_common_dir:ro" \
        --volume "$tilefoundry_root:/workspace/tilefoundry:ro" \
        --volume "$repo_dir:/workspace/tileops-foundry-loop:ro" \
        --volume "$cache_root:/ci-cache" \
        --workdir /workspace/tileops \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env GIT_OPTIONAL_LOCKS=0 \
        --env PYTHONUNBUFFERED=1 \
        "$image" sleep infinity >/dev/null

    docker exec --workdir /workspace/tileops "$container_name" \
        python -m pip install --quiet --root-user-action=ignore \
        --no-deps --no-build-isolation --editable /workspace/tileops
    docker exec "$container_name" python -c 'import cupti'
}

action=${1:-shell}
case "$action" in
    destroy)
        destroy_container
        printf '%s removed\n' "$container_name"
        ;;
    recreate)
        destroy_container
        ensure_container
        printf '%s GPU=%s worktree=%s\n' "$container_name" "$gpu" "$worktree"
        ;;
    start)
        ensure_container
        printf '%s GPU=%s worktree=%s\n' "$container_name" "$gpu" "$worktree"
        ;;
    name)
        ensure_container
        printf '%s\n' "$container_name"
        ;;
    shell)
        ensure_container
        exec_args=(-i)
        if [[ -t 0 && -t 1 ]]; then
            exec_args+=(-t)
        fi
        exec docker exec "${exec_args[@]}" --workdir /workspace/tileops "$container_name" bash
        ;;
    exec)
        shift
        (( $# > 0 )) || { echo "usage: $0 exec COMMAND [ARG...]" >&2; exit 2; }
        ensure_container
        exec docker exec --workdir /workspace/tileops "$container_name" "$@"
        ;;
    *)
        ensure_container
        exec docker exec --workdir /workspace/tileops "$container_name" "$@"
        ;;
esac
