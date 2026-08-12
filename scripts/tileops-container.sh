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
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
wheel_root=${TILEFOUNDRY_WHEEL_ROOT:-$cache_root/tilefoundry-wheel}
loop_state_root=${TILEOPS_LOOP_STATE_ROOT:-$repo_dir/rounds}
if [[ -f "$wheel_root/current.env" ]]; then
    # shellcheck disable=SC1090
    source "$wheel_root/current.env"
fi
tilefoundry_wheel=${TILEFOUNDRY_WHEEL:-}
tilefoundry_wheel_sha256=${TILEFOUNDRY_WHEEL_SHA256:-}
tilefoundry_wheel_commit=${TILEFOUNDRY_WHEEL_COMMIT:-}
tilefoundry_requirements_sha256=${TILEFOUNDRY_REQUIREMENTS_SHA256:-}

if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then
    "$TILEOPS_DOCKER_BOOTSTRAP"
fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then
    export DOCKER_HOST=$TILEOPS_DOCKER_HOST
fi
image_library_path=$(docker image inspect "$image" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
    | sed -n 's/^LD_LIBRARY_PATH=//p')
container_library_path="/usr/local/lib${image_library_path:+:$image_library_path}"

worktree=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ ! -f "$worktree/src/tileops/__init__.py" ]]; then
    worktree=$default_tileops
fi
if [[ -z "$worktree" || ! -f "$worktree/pyproject.toml" ]]; then
    echo "Run from a TileOPs worktree or set TILEOPS_REPO" >&2
    exit 1
fi
if [[ -z "$tilefoundry_wheel" || ! -f "$tilefoundry_wheel" ]]; then
    echo "No admitted TileFoundry wheel; run scripts/build_tilefoundry_wheel.sh" >&2
    exit 1
fi
actual_wheel_sha256=$(sha256sum "$tilefoundry_wheel" | awk '{print $1}')
if [[ "$actual_wheel_sha256" != "$tilefoundry_wheel_sha256" ]]; then
    echo "TileFoundry wheel SHA-256 does not match admission record" >&2
    exit 1
fi
dependency_root="$wheel_root/deps/$tilefoundry_requirements_sha256"
if [[ -z "$tilefoundry_requirements_sha256" || ! -f "$dependency_root/.complete" ]]; then
    echo "TileFoundry runtime dependency bundle is not admitted" >&2
    exit 1
fi
container_wheel="/opt/tilefoundry-wheel/$tilefoundry_wheel_commit/$(basename -- "$tilefoundry_wheel")"
container_dependencies="/opt/tilefoundry-wheel/deps/$tilefoundry_requirements_sha256"

git_common_dir=$(git -C "$worktree" rev-parse --path-format=absolute --git-common-dir)
mkdir -p "$cache_root" "$loop_state_root"
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
        actual_wheel=$(docker container inspect \
            --format '{{index .Config.Labels "tileops-foundry-loop.tilefoundry-wheel-sha256"}}' "$container_name")
        actual_requirements=$(docker container inspect \
            --format '{{index .Config.Labels "tileops-foundry-loop.tilefoundry-requirements-sha256"}}' "$container_name")
        [[ "$mounted_root" == "$worktree" ]] || {
            echo "$container_name belongs to $mounted_root" >&2
            exit 1
        }
        [[ "$actual_requirements" == "$tilefoundry_requirements_sha256" ]] || {
            echo "$container_name has a stale TileFoundry dependency bundle; run: $0 recreate" >&2
            exit 1
        }
        [[ "$actual_image" == "$expected_image" && "$actual_schema" == "$env_schema" \
            && "$actual_wheel" == "$tilefoundry_wheel_sha256" ]] || {
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
    else
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
            --label "tileops-foundry-loop.tilefoundry-wheel-sha256=$tilefoundry_wheel_sha256" \
            --label "tileops-foundry-loop.tilefoundry-commit=$tilefoundry_wheel_commit" \
            --label "tileops-foundry-loop.tilefoundry-requirements-sha256=$tilefoundry_requirements_sha256" \
            --device "nvidia.com/gpu=$gpu" --ipc=host --shm-size=16g \
            --volume "$worktree:/workspace/tileops" \
            --volume "$git_common_dir:$git_common_dir:ro" \
            --volume "$wheel_root:/opt/tilefoundry-wheel:ro" \
            --volume "$repo_dir:/workspace/tileops-foundry-loop:ro" \
            --volume "$loop_state_root:/workspace/tileops-loop-state" \
            --volume "$cache_root:/ci-cache" \
            --workdir /workspace/tileops \
            --env CUDA_VISIBLE_DEVICES=0 \
            --env GIT_OPTIONAL_LOCKS=0 \
            --env PYTHONUNBUFFERED=1 \
            --env "TILEFOUNDRY_WHEEL_COMMIT=$tilefoundry_wheel_commit" \
            --env "TILEFOUNDRY_WHEEL_SHA256=$tilefoundry_wheel_sha256" \
            --env "LD_LIBRARY_PATH=$container_library_path" \
            "$image" sleep infinity >/dev/null
    fi

    marker="/opt/tileops-env-${env_schema}-${tilefoundry_wheel_sha256:0:12}-${tilefoundry_requirements_sha256:0:12}"
    if ! docker exec "$container_name" test -f "$marker"; then
        docker exec --workdir /workspace/tileops "$container_name" \
            python -m pip install --quiet --root-user-action=ignore \
            --no-deps --no-build-isolation --editable /workspace/tileops
        docker exec "$container_name" bash -lc \
            "python -m pip install --quiet --root-user-action=ignore --no-deps '$container_dependencies'/*.whl"
        docker exec "$container_name" python -m pip install --quiet \
            --root-user-action=ignore --no-deps --force-reinstall "$container_wheel"
        docker exec "$container_name" python -c \
            'import cupti, pathlib, tilefoundry; p=pathlib.Path(tilefoundry.__file__).resolve(); assert "/workspace/tilefoundry" not in str(p), p'
        docker exec "$container_name" python -c \
            'import importlib.metadata as m; from packaging.requirements import Requirement; from packaging.version import Version; reqs=[Requirement(x) for x in m.requires("tilefoundry") or ()]; bad=[str(r) for r in reqs if (not r.marker or r.marker.evaluate()) and Version(m.version(r.name)) not in r.specifier]; assert not bad, bad'
        docker exec "$container_name" tilefoundry --help >/dev/null
        docker exec "$container_name" touch "$marker"
    fi
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
