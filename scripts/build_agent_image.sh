#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then source "$repo_dir/.env"; fi
if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then "$TILEOPS_DOCKER_BOOTSTRAP"; fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then export DOCKER_HOST=$TILEOPS_DOCKER_HOST; fi

docker image inspect "$TILEOPS_RUNNER_IMAGE" >/dev/null 2>&1 || docker pull "$TILEOPS_RUNNER_IMAGE"
base_id=$(docker image inspect "$TILEOPS_RUNNER_IMAGE" --format '{{.Id}}')
context_id=$(sha256sum \
    "$repo_dir/container/Dockerfile" \
    "$repo_dir/container/entrypoint.sh" \
    "$repo_dir/container/sshd_config" | sha256sum | awk '{print $1}')
base_id=${base_id#sha256:}
tag="tileops-foundry-loop:ssh-${base_id:0:16}-${context_id:0:16}"
if ! docker image inspect "$tag" >/dev/null 2>&1; then
    docker build --quiet \
        --build-arg "BASE_IMAGE=$TILEOPS_RUNNER_IMAGE" \
        --tag "$tag" "$repo_dir/container"
fi

printf '%s\n' "$tag"
