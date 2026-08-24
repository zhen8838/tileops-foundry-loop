#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then source "$repo_dir/.env"; fi
if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then "$TILEOPS_DOCKER_BOOTSTRAP"; fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then export DOCKER_HOST=$TILEOPS_DOCKER_HOST; fi

worktree=$(git rev-parse --show-toplevel 2>/dev/null || true)
[[ -n "$worktree" && -f "$worktree/pyproject.toml" ]] || {
    echo "run from a TileOPs worktree" >&2
    exit 2
}
task=${FOREMAN_TASK:?FOREMAN_TASK is required}
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
worktree_key=$(printf '%s' "$worktree" | sha256sum | awk '{print substr($1,1,10)}')
container_name="tileops-round-${task}-${worktree_key}"
action=${1:-start}
if [[ "$action" == destroy ]]; then
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    printf '%s removed\n' "$container_name"
    exit 0
fi

admission="$cache_root/worker-admissions/$task.env"
[[ -f "$admission" ]] || { echo "missing worker admission: $admission" >&2; exit 1; }
source "$admission"
round_host=${TILEOPS_ROUND_HOST:?TILEOPS_ROUND_HOST is required}
[[ -d "$round_host" ]] || { echo "round directory does not exist: $round_host" >&2; exit 1; }

wheel_root=${TILEFOUNDRY_WHEEL_ROOT:-$cache_root/tilefoundry-wheel}
[[ -f "$wheel_root/current.env" ]] || { echo "missing admitted wheel record" >&2; exit 1; }
source "$wheel_root/current.env"
[[ -f "$TILEFOUNDRY_WHEEL" ]] || { echo "admitted wheel is missing: $TILEFOUNDRY_WHEEL" >&2; exit 1; }
actual_wheel_sha256=$(sha256sum "$TILEFOUNDRY_WHEEL" | awk '{print $1}')
[[ "$actual_wheel_sha256" == "$TILEFOUNDRY_WHEEL_SHA256" ]] || {
    echo "admitted wheel SHA-256 mismatch" >&2
    exit 1
}
wheel_name=$(basename -- "$TILEFOUNDRY_WHEEL")
container_wheel="/opt/tilefoundry-wheel/$TILEFOUNDRY_WHEEL_COMMIT/$wheel_name"
container_deps="/opt/tilefoundry-wheel/deps/$TILEFOUNDRY_REQUIREMENTS_SHA256"
[[ -f "$wheel_root/deps/$TILEFOUNDRY_REQUIREMENTS_SHA256/.complete" ]] || {
    echo "admitted TileFoundry dependency bundle is incomplete" >&2
    exit 1
}
agent_image=$("$repo_dir/scripts/build_agent_image.sh")

ssh_dir="$cache_root/round-ssh/$task"
tilelang_cache="$cache_root/runtime-cache/tilelang"
triton_cache="$cache_root/runtime-cache/triton"
mkdir -p "$ssh_dir"
mkdir -p "$tilelang_cache" "$triton_cache"
chmod 0700 "$ssh_dir"
if [[ ! -f "$ssh_dir/id_ed25519" ]]; then
    ssh-keygen -q -t ed25519 -N "" -f "$ssh_dir/id_ed25519"
fi
public_key=$(cat "$ssh_dir/id_ed25519.pub")

container_exists=false
requested_gpu=${TILEOPS_GPU:-}
if docker container inspect "$container_name" >/dev/null 2>&1; then container_exists=true; fi
if $container_exists; then
    current_worktree=$(docker inspect --format '{{index .Config.Labels "tileops.worktree"}}' "$container_name")
    current_round=$(docker inspect --format '{{index .Config.Labels "tileops.round"}}' "$container_name")
    current_wheel=$(docker inspect --format '{{index .Config.Labels "tileops.wheel"}}' "$container_name")
    current_requirements=$(docker inspect --format '{{index .Config.Labels "tileops.requirements"}}' "$container_name")
    current_image=$(docker inspect --format '{{index .Config.Labels "tileops.image"}}' "$container_name")
    gpu=$(docker inspect --format '{{index .Config.Labels "tileops.gpu"}}' "$container_name")
    if [[ "$current_worktree" != "$worktree" || "$current_round" != "$round_host" || \
        "$current_wheel" != "$TILEFOUNDRY_WHEEL_SHA256" || \
        "$current_requirements" != "$TILEFOUNDRY_REQUIREMENTS_SHA256" || \
        "$current_image" != "$agent_image" || \
        ( -n "$requested_gpu" && "$gpu" != "$requested_gpu" ) ]]; then
        docker rm -f "$container_name" >/dev/null
        container_exists=false
    fi
fi
if ! $container_exists; then
    gpu=$requested_gpu
    if [[ -z "$gpu" ]]; then
        gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | tr -d ' ' | tr ',' ' ' | sort -k2,2n -k1,1n | head -n1 | awk '{print $1}')
    fi
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "TILEOPS_GPU must be numeric" >&2; exit 1; }
    docker run --detach --init --name "$container_name" \
        --label "tileops.worktree=$worktree" \
        --label "tileops.round=$round_host" \
        --label "tileops.wheel=$TILEFOUNDRY_WHEEL_SHA256" \
        --label "tileops.requirements=$TILEFOUNDRY_REQUIREMENTS_SHA256" \
        --label "tileops.image=$agent_image" \
        --label "tileops.gpu=$gpu" \
        --device "nvidia.com/gpu=$gpu" --ipc=host --shm-size=16g \
        --publish 127.0.0.1::22 \
        --volume "$worktree:/workspace/tileops" \
        --volume "$round_host:/workspace/round" \
        --volume "$wheel_root:/opt/tilefoundry-wheel:ro" \
        --volume "$tilelang_cache:/ci-cache/tilelang" \
        --volume "$triton_cache:/ci-cache/triton" \
        --workdir /workspace/round \
        --env "TILEOPS_SSH_PUBLIC_KEY=$public_key" \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env GIT_OPTIONAL_LOCKS=0 \
        --env PYTHONUNBUFFERED=1 \
        --env "TILEOPS_PHYSICAL_GPU=$gpu" \
        --env TILELANG_CACHE_DIR=/ci-cache/tilelang \
        --env TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
        --env TRITON_CACHE_DIR=/ci-cache/triton \
        "$agent_image" >/dev/null
fi

if [[ $(docker inspect --format '{{.State.Running}}' "$container_name") != true ]]; then
    docker start "$container_name" >/dev/null
fi

marker="/var/lib/tileops-wheel-$TILEFOUNDRY_WHEEL_SHA256"
if ! docker exec "$container_name" test -f "$marker"; then
    docker exec "$container_name" python -m pip install --quiet --root-user-action=ignore \
        --no-deps "$container_wheel"
    if docker exec "$container_name" test -f "$container_deps/.complete"; then
        docker exec "$container_name" bash -lc \
            "python -m pip install --quiet --root-user-action=ignore --no-deps '$container_deps'/*.whl"
    fi
    docker exec --workdir /workspace/tileops "$container_name" \
        python -m pip install --quiet --root-user-action=ignore --no-deps --editable .
    docker exec "$container_name" python -c \
        'import pathlib, tilefoundry; p=pathlib.Path(tilefoundry.__file__).resolve(); assert "/workspace/tilefoundry" not in str(p), p'
    docker exec "$container_name" mkdir -p /var/lib
    docker exec "$container_name" touch "$marker"
fi

port=$(docker port "$container_name" 22/tcp | sed -n 's/.*:\([0-9][0-9]*\)$/\1/p' | head -n1)
[[ "$port" =~ ^[0-9]+$ ]] || { echo "could not determine SSH port" >&2; exit 1; }

tmp=$(mktemp "$admission.XXXXXX")
{
    printf 'export TILEOPS_ROUND_HOST=%q\n' "$round_host"
    printf 'export TILEOPS_ROUND_SLUG=%q\n' "$TILEOPS_ROUND_SLUG"
    printf 'export TILEOPS_TASK=%q\n' "$task"
    printf 'export TILEOPS_WORKER_CONTAINER=%q\n' "$container_name"
    printf 'export TILEOPS_SSH_IDENTITY=%q\n' "$ssh_dir/id_ed25519"
    printf 'export TILEOPS_SSH_TARGET=%q\n' "root@127.0.0.1:${port}:/workspace/round"
    printf 'export TILEOPS_PI_SSH_TARGET=%q\n' "root@127.0.0.1:${port}:/workspace/round"
    printf 'export TILEOPS_PI_SSH_EXTENSION=%q\n' "$repo_dir/integrations/pi/ssh.ts"
    printf 'export TILEOPS_AGENT_IMAGE=%q\n' "$agent_image"
    printf 'export TILEOPS_PHYSICAL_GPU=%q\n' "${gpu:-${TILEOPS_GPU:-}}"
} >"$tmp"
chmod 600 "$tmp"
mv -- "$tmp" "$admission"

{
    printf '# Container Environment\n\n'
    printf -- '- image: `%s`\n' "$agent_image"
    printf -- '- container: `%s`\n' "$container_name"
    printf -- '- GPU: `%s`\n' "${gpu:-${TILEOPS_GPU:-auto}}"
    printf -- '- TileFoundry wheel commit: `%s`\n' "$TILEFOUNDRY_WHEEL_COMMIT"
    printf -- '- TileFoundry wheel SHA-256: `%s`\n' "$TILEFOUNDRY_WHEEL_SHA256"
    printf -- '- round mount: `/workspace/round`\n'
    printf -- '- TileOPs mount: `/workspace/tileops`\n'
    printf -- '- SSH target: `root@127.0.0.1:%s:/workspace/round`\n' "$port"
} >"$round_host/environment.md"

case "$action" in
    start) printf '%s GPU=%s SSH=%s\n' "$container_name" "${gpu:-${TILEOPS_GPU:-auto}}" "root@127.0.0.1:${port}" ;;
    name) printf '%s\n' "$container_name" ;;
    port) printf '%s\n' "$port" ;;
    exec) shift; (( $# > 0 )) || { echo "usage: $0 exec COMMAND [ARG...]" >&2; exit 2; }; exec docker exec --workdir /workspace/round "$container_name" "$@" ;;
    shell) exec docker exec -it --workdir /workspace/round "$container_name" bash ;;
    *) echo "usage: $0 [start|name|port|exec|shell|destroy]" >&2; exit 2 ;;
esac
