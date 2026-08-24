#!/usr/bin/env bash

script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
repo_dir=$(cd -- "$(dirname -- "$script_path")/.." && pwd)

# Foreman sources this file so the exported SSH target and round cwd remain in
# the pane where Pi starts. Heavy setup runs in a child shell with strict mode.
if [[ ${1:-} != --prepare ]]; then
    [[ ${BASH_SOURCE[0]} != "$0" ]] || {
        echo "setup-worktree.sh must be sourced by Foreman" >&2
        exit 2
    }
    worker_env=$(bash "$script_path" --prepare) || return
    source "$worker_env" || return
    export TILEOPS_ROUND_HOST TILEOPS_ROUND_SLUG TILEOPS_WORKER_CONTAINER
    export TILEOPS_SSH_IDENTITY TILEOPS_PI_SSH_TARGET TILEOPS_AGENT_IMAGE
    cd "$TILEOPS_ROUND_HOST" || return
    return
fi

set -Eeuo pipefail
stage=initialization
report_failure() {
    local rc=${1:-$?}
    if (( rc != 0 )); then
        printf 'tileops setup-worktree: [%s] failed (exit %s)\n' "$stage" "$rc" >&2
        printf '  task=%s\n  worktree=%s\n' \
            "${FOREMAN_TASK:-<unset>}" "${FOREMAN_WORKTREE:-<unset>}" >&2
    fi
}
trap report_failure EXIT

if [[ -f "$repo_dir/.env" ]]; then source "$repo_dir/.env"; fi

stage='configuration validation'
: "${FOREMAN_WORKTREE:?Foreman did not provide FOREMAN_WORKTREE}"
: "${FOREMAN_TASK:?Foreman did not provide FOREMAN_TASK}"
: "${FOREMAN_PROMPT:?Use foreman assign --prompt to describe the round}"
: "${FOREMAN_AGENT_ARGS_FILE:?Foreman did not provide FOREMAN_AGENT_ARGS_FILE}"
: "${TILEFOUNDRY_REPO:?Set TILEFOUNDRY_REPO in .env}"

agent_image=${TILEOPS_AGENT_IMAGE:-tileops-foundry-loop:agent}
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
wheel_root=${TILEFOUNDRY_WHEEL_ROOT:-$cache_root/tilefoundry-wheel}
host_uv_bin=${TILEFOUNDRY_UV_BIN:-$(command -v uv || true)}
if [[ ! -x "$host_uv_bin" ]]; then
    for candidate in "$HOME/bin/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [[ -x "$candidate" ]]; then
            host_uv_bin=$candidate
            break
        fi
    done
fi
[[ -x "$host_uv_bin" ]] || {
    echo "uv is not installed; rerun $repo_dir/setup or set TILEFOUNDRY_UV_BIN" >&2
    exit 1
}
pi_command=$(command -v pi || true)
[[ -n "$pi_command" ]] || {
    echo "pi is not installed; rerun $repo_dir/setup" >&2
    exit 1
}
pi_cli=$(readlink -f -- "$pi_command")
pi_root=$(cd -- "$(dirname -- "$pi_cli")/.." && pwd)
pi_ssh_extension=${PI_SSH_EXTENSION:-$pi_root/examples/extensions/ssh.ts}
[[ -f "$pi_ssh_extension" ]] || {
    echo "Pi SSH extension not found: $pi_ssh_extension; rerun $repo_dir/setup" >&2
    exit 1
}

for command in docker gh git nvidia-smi python3 ssh ssh-keygen; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required host command is missing: $command" >&2
        exit 1
    }
done

if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then "$TILEOPS_DOCKER_BOOTSTRAP"; fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then export DOCKER_HOST=$TILEOPS_DOCKER_HOST; fi
docker image inspect "$agent_image" >/dev/null 2>&1 || {
    echo "missing $agent_image; run $repo_dir/setup first" >&2
    exit 1
}

build_tilefoundry_wheel() {
    local requirements requirements_sha256 tilefoundry_repo commit remote
    local uv_bin builder_root python_bin destination temp_root source_tree staging
    local dependency_root temp_dependencies

    tilefoundry_repo=$TILEFOUNDRY_REPO
    requirements="$repo_dir/tilefoundry-requirements.txt"
    requirements_sha256=$(sha256sum "$requirements" | awk '{print $1}')
    mkdir -p "$wheel_root"
    exec 9>"$wheel_root/.build.lock"
    flock 9

    if [[ -n ${TILEFOUNDRY_COMMIT:-} ]]; then
        commit=$TILEFOUNDRY_COMMIT
    else
        remote=upstream
        git -C "$tilefoundry_repo" remote | grep -qx upstream || remote=origin
        git -C "$tilefoundry_repo" fetch --quiet "$remote" main
        commit=$(git -C "$tilefoundry_repo" rev-parse "$remote/main")
    fi
    commit=$(git -C "$tilefoundry_repo" rev-parse "$commit^{commit}")

    uv_bin=$host_uv_bin
    builder_root="$wheel_root/builder"
    python_bin="$builder_root/bin/python"
    if [[ ! -x "$python_bin" ]]; then
        "$uv_bin" venv --seed --no-project \
            --python "${TILEFOUNDRY_BUILD_PYTHON:-3.12}" "$builder_root" >&2
        "$uv_bin" pip install --python "$python_bin" \
            'setuptools>=68' 'setuptools-scm>=8' >&2
    fi

    destination="$wheel_root/$commit"
    mkdir -p "$destination"
    tilefoundry_wheel=$(find "$destination" -maxdepth 1 -type f \
        -name 'tilefoundry-*.whl' -print -quit)
    if [[ -z "$tilefoundry_wheel" ]]; then
        temp_root=$(mktemp -d "${TMPDIR:-/tmp}/tilefoundry-wheel.XXXXXX")
        source_tree="$temp_root/source"
        cleanup_wheel() {
            git -C "$tilefoundry_repo" worktree remove --force "$source_tree" \
                >/dev/null 2>&1 || true
            rm -rf -- "$temp_root"
        }
        trap 'rc=$?; cleanup_wheel; report_failure "$rc"' EXIT
        git -C "$tilefoundry_repo" worktree add --detach "$source_tree" "$commit" >&2
        staging="$temp_root/wheel"
        mkdir -p "$staging"
        "$python_bin" -m pip wheel "$source_tree" --no-deps --no-build-isolation \
            --wheel-dir "$staging" >&2
        tilefoundry_wheel=$(find "$staging" -maxdepth 1 -type f \
            -name 'tilefoundry-*.whl' -print -quit)
        [[ -n "$tilefoundry_wheel" ]] || {
            echo "TileFoundry wheel build produced no wheel" >&2
            exit 1
        }
        cp -- "$tilefoundry_wheel" "$destination/"
        tilefoundry_wheel="$destination/$(basename -- "$tilefoundry_wheel")"
        cleanup_wheel
        trap report_failure EXIT
    else
        tilefoundry_wheel=$(realpath "$tilefoundry_wheel")
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

    tilefoundry_wheel=$(realpath "$tilefoundry_wheel")
    tilefoundry_wheel_sha=$(sha256sum "$tilefoundry_wheel" | awk '{print $1}')
    tilefoundry_commit=$commit
    tilefoundry_requirements_sha=$requirements_sha256
}

create_round() {
    round_host="$repo_dir/rounds/$FOREMAN_TASK"
    round_marker="$round_host/.task-sha256"
    task_sha=$(printf '%s\n%s\n' "$FOREMAN_PROMPT" "${FOREMAN_BRANCH:-}" \
        | sha256sum | awk '{print $1}')

    if [[ -e "$round_host" ]]; then
        [[ -f "$round_marker" && $(<"$round_marker") == "$task_sha" ]] || {
            echo "round already exists for different input: $round_host" >&2
            exit 1
        }
        return
    fi

    cp -a "$repo_dir/template" "$round_host"
    printf '%s\n' "$task_sha" >"$round_marker"
    ROUND_TASK=$FOREMAN_TASK \
    ROUND_PROMPT=$FOREMAN_PROMPT \
    ROUND_BRANCH=${FOREMAN_BRANCH:-} \
    ROUND_TILEOPS_BASE=$(git -C "$FOREMAN_WORKTREE" rev-parse HEAD) \
    ROUND_TILEFOUNDRY_COMMIT=$tilefoundry_commit \
        python3 - "$round_host/brief.md" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
values = {
    "TASK": os.environ["ROUND_TASK"],
    "PROMPT": os.environ["ROUND_PROMPT"],
    "BRANCH": os.environ["ROUND_BRANCH"],
    "TILEOPS_BASE": os.environ["ROUND_TILEOPS_BASE"],
    "TILEFOUNDRY_COMMIT": os.environ["ROUND_TILEFOUNDRY_COMMIT"],
}
for key, value in values.items():
    text = text.replace("{{" + key + "}}", value)
path.write_text(text)
PY
}

start_container() {
    local worktree worktree_key container_name wheel_name container_wheel container_deps
    local ssh_dir tilelang_cache triton_cache public_key requested_gpu gpu
    local git_common_dir host_git_config host_gh_config host_gh_bin host_ssh_dir auth_key
    local container_exists current_worktree current_round current_wheel current_image current_auth
    local marker port ssh_ready tmp image_id ssh_alias ssh_config_dir ssh_config
    local ssh_include pi_ssh_target args_tmp
    local -a auth_mounts auth_env

    worktree=$(cd -- "$FOREMAN_WORKTREE" && pwd -P)
    worktree_key=$(printf '%s' "$worktree" | sha256sum | awk '{print substr($1,1,10)}')
    container_name="tileops-round-${FOREMAN_TASK}-${worktree_key}"
    wheel_name=$(basename -- "$tilefoundry_wheel")
    container_wheel="/opt/tilefoundry-wheel/$tilefoundry_commit/$wheel_name"
    container_deps="/opt/tilefoundry-wheel/deps/$tilefoundry_requirements_sha"
    image_id=$(docker image inspect "$agent_image" --format '{{.Id}}')

    ssh_dir="$cache_root/round-ssh/$FOREMAN_TASK"
    tilelang_cache="$cache_root/runtime-cache/tilelang"
    triton_cache="$cache_root/runtime-cache/triton"
    mkdir -p "$ssh_dir" "$tilelang_cache" "$triton_cache"
    chmod 0700 "$ssh_dir"
    if [[ ! -f "$ssh_dir/id_ed25519" ]]; then
        ssh-keygen -q -t ed25519 -N "" -f "$ssh_dir/id_ed25519"
    fi
    public_key=$(<"$ssh_dir/id_ed25519.pub")

    git_common_dir=$(git -C "$worktree" rev-parse --path-format=absolute --git-common-dir)
    host_git_config=${TILEOPS_HOST_GIT_CONFIG:-$HOME/.gitconfig}
    host_gh_config=${TILEOPS_HOST_GH_CONFIG:-$HOME/.config/gh}
    host_gh_bin=${TILEOPS_HOST_GH_BIN:-$(command -v gh || true)}
    host_ssh_dir=${TILEOPS_HOST_SSH_DIR:-$HOME/.ssh}
    [[ -f "$host_git_config" ]] || host_git_config=
    [[ -d "$host_gh_config" ]] || host_gh_config=
    [[ -x "$host_gh_bin" ]] || host_gh_bin=
    [[ -d "$host_ssh_dir" ]] || host_ssh_dir=

    auth_mounts=(--volume "$git_common_dir:$git_common_dir")
    auth_env=(--env GIT_CONFIG_GLOBAL=/root/.gitconfig)
    if [[ -n "$host_git_config" ]]; then
        auth_mounts+=(--volume "$host_git_config:/opt/tileops-host-auth/gitconfig:ro")
    fi
    if [[ -n "$host_gh_config" && -n "$host_gh_bin" ]]; then
        auth_mounts+=(
            --volume "$host_gh_config:/opt/tileops-host-auth/gh:ro"
            --volume "$host_gh_bin:/usr/local/bin/gh:ro"
        )
        auth_env+=(--env GH_CONFIG_DIR=/opt/tileops-host-auth/gh)
    fi
    if [[ -n "$host_ssh_dir" ]]; then
        auth_mounts+=(--volume "$host_ssh_dir:$host_ssh_dir:ro")
        auth_env+=(--env "TILEOPS_HOST_SSH_DIR=$host_ssh_dir")
    fi
    auth_key=$(printf '%s\n' "$git_common_dir" "$host_git_config" \
        "$host_gh_config" "$host_gh_bin" "$host_ssh_dir" | sha256sum | awk '{print $1}')

    container_exists=false
    requested_gpu=${TILEOPS_GPU:-}
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        container_exists=true
    fi
    if $container_exists; then
        current_worktree=$(docker inspect --format \
            '{{index .Config.Labels "tileops.worktree"}}' "$container_name")
        current_round=$(docker inspect --format \
            '{{index .Config.Labels "tileops.round"}}' "$container_name")
        current_wheel=$(docker inspect --format \
            '{{index .Config.Labels "tileops.wheel"}}' "$container_name")
        current_image=$(docker inspect --format \
            '{{index .Config.Labels "tileops.image"}}' "$container_name")
        current_auth=$(docker inspect --format \
            '{{index .Config.Labels "tileops.auth"}}' "$container_name")
        gpu=$(docker inspect --format '{{index .Config.Labels "tileops.gpu"}}' "$container_name")
        if [[ "$current_worktree" != "$worktree" || "$current_round" != "$round_host" || \
            "$current_wheel" != "$tilefoundry_wheel_sha" || \
            "$current_image" != "$image_id" || "$current_auth" != "$auth_key" || \
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
            --label "tileops.task=$FOREMAN_TASK" \
            --label "tileops.worktree=$worktree" \
            --label "tileops.round=$round_host" \
            --label "tileops.wheel=$tilefoundry_wheel_sha" \
            --label "tileops.image=$image_id" \
            --label "tileops.auth=$auth_key" \
            --label "tileops.gpu=$gpu" \
            --device "nvidia.com/gpu=$gpu" --ipc=host --shm-size=16g \
            --publish 127.0.0.1::22 \
            --volume "$worktree:/workspace/tileops" \
            --volume "$round_host:/workspace/round" \
            --volume "$wheel_root:/opt/tilefoundry-wheel:ro" \
            --volume "$tilelang_cache:/ci-cache/tilelang" \
            --volume "$triton_cache:/ci-cache/triton" \
            "${auth_mounts[@]}" \
            --workdir /workspace/round \
            --env "TILEOPS_SSH_PUBLIC_KEY=$public_key" \
            --env CUDA_VISIBLE_DEVICES=0 \
            --env GIT_OPTIONAL_LOCKS=0 \
            --env PYTHONUNBUFFERED=1 \
            --env "TILEOPS_PHYSICAL_GPU=$gpu" \
            --env TILELANG_CACHE_DIR=/ci-cache/tilelang \
            --env TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
            --env TRITON_CACHE_DIR=/ci-cache/triton \
            "${auth_env[@]}" \
            "$agent_image" >/dev/null
    fi

    if [[ $(docker inspect --format '{{.State.Running}}' "$container_name") != true ]]; then
        docker start "$container_name" >/dev/null
    fi

    marker="/var/lib/tileops-wheel-$tilefoundry_wheel_sha"
    if ! docker exec "$container_name" test -f "$marker"; then
        docker exec "$container_name" python -m pip install --quiet \
            --root-user-action=ignore --no-deps "$container_wheel"
        docker exec "$container_name" bash -lc \
            "python -m pip install --quiet --root-user-action=ignore --no-deps '$container_deps'/*.whl"
        docker exec --workdir /workspace/tileops "$container_name" \
            python -m pip install --quiet --root-user-action=ignore --no-deps --editable .
        docker exec "$container_name" python -c \
            'import pathlib, tilefoundry; p=pathlib.Path(tilefoundry.__file__).resolve(); assert "/workspace/tilefoundry" not in str(p), p'
        docker exec "$container_name" touch "$marker"
    fi

    port=$(docker port "$container_name" 22/tcp \
        | sed -n 's/.*:\([0-9][0-9]*\)$/\1/p' | head -n1)
    [[ "$port" =~ ^[0-9]+$ ]] || { echo "could not determine SSH port" >&2; exit 1; }
    ssh_alias="tileops-${FOREMAN_TASK}-${worktree_key}"
    ssh_config_dir="$cache_root/ssh-config"
    ssh_include="Include \"$ssh_config_dir/*\""
    [[ -f "$HOME/.ssh/config" ]] && grep -Fqx "$ssh_include" "$HOME/.ssh/config" || {
        echo "SSH config does not include $ssh_config_dir; rerun $repo_dir/setup" >&2
        exit 1
    }
    mkdir -p "$ssh_config_dir"
    ssh_config="$ssh_config_dir/$ssh_alias"
    tmp=$(mktemp "$ssh_config.XXXXXX")
    {
        printf 'Host %s\n' "$ssh_alias"
        printf '    HostName 127.0.0.1\n'
        printf '    User root\n'
        printf '    Port %s\n' "$port"
        printf '    IdentityFile %s\n' "$ssh_dir/id_ed25519"
        printf '    IdentitiesOnly yes\n'
        printf '    BatchMode yes\n'
        printf '    StrictHostKeyChecking no\n'
        printf '    UserKnownHostsFile /dev/null\n'
    } >"$tmp"
    chmod 0600 "$tmp"
    mv -- "$tmp" "$ssh_config"

    ssh_ready=false
    for _ in {1..50}; do
        if ssh "$ssh_alias" \
            'test "$CUDA_HOME" = /usr/local/cuda &&
             test "$(command -v nvcc)" = /usr/local/cuda/bin/nvcc &&
             test "$TILELANG_CACHE_DIR" = /ci-cache/tilelang &&
             git -C /workspace/tileops status --short >/dev/null' >/dev/null 2>&1; then
            ssh_ready=true
            break
        fi
        sleep 0.1
    done
    $ssh_ready || { echo "round SSH environment did not become ready" >&2; exit 1; }

    if [[ -n "$host_gh_config" && -n "$host_gh_bin" ]]; then
        ssh "$ssh_alias" \
            'gh auth status --hostname github.com >/dev/null 2>&1 &&
             printf "protocol=https\nhost=github.com\n\n" | git credential fill | grep -q "^password="' || {
            echo "mounted GitHub credentials are not usable" >&2
            exit 1
        }
    fi

    worker_env="$round_host/.worker-env"
    tmp=$(mktemp "$worker_env.XXXXXX")
    {
        printf 'export TILEOPS_ROUND_HOST=%q\n' "$round_host"
        printf 'export TILEOPS_ROUND_SLUG=%q\n' "$FOREMAN_TASK"
        printf 'export TILEOPS_WORKER_CONTAINER=%q\n' "$container_name"
        printf 'export TILEOPS_SSH_IDENTITY=%q\n' "$ssh_dir/id_ed25519"
        pi_ssh_target="$ssh_alias:/workspace/round"
        printf 'export TILEOPS_PI_SSH_TARGET=%q\n' "$pi_ssh_target"
        printf 'export TILEOPS_AGENT_IMAGE=%q\n' "$agent_image"
    } >"$tmp"
    chmod 0600 "$tmp"
    mv -- "$tmp" "$worker_env"

    {
        printf '# Environment\n\n'
        printf -- '- container: `%s`\n' "$container_name"
        printf -- '- image: `%s`\n' "$image_id"
        printf -- '- GPU: `%s`\n' "$gpu"
        printf -- '- TileFoundry wheel: `%s`\n' "$tilefoundry_commit"
        printf -- '- TileOPs: `/workspace/tileops`\n'
        printf -- '- round: `/workspace/round`\n'
        printf -- '- SSH: `root@127.0.0.1:%s`\n' "$port"
    } >"$round_host/environment.md"

    args_tmp=$(mktemp "$FOREMAN_AGENT_ARGS_FILE.XXXXXX")
    python3 - "$args_tmp" "$pi_ssh_extension" "$pi_ssh_target" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(["-e", sys.argv[2], "--ssh", sys.argv[3]], stream)
    stream.write("\n")
PY
    mv -- "$args_tmp" "$FOREMAN_AGENT_ARGS_FILE"
}

stage='TileFoundry wheel'
build_tilefoundry_wheel
stage='round workspace'
create_round
stage='container, SSH, and agent handoff'
start_container
stage=complete
printf '%s\n' "$worker_env"
trap - EXIT
