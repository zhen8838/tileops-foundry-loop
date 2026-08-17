#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo_dir/config/defaults.env"
if [[ -f "$repo_dir/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_dir/.env"
fi
if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then
    "$TILEOPS_DOCKER_BOOTSTRAP"
fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then
    export DOCKER_HOST=$TILEOPS_DOCKER_HOST
fi

image=${TILEOPS_RUNNER_IMAGE:?TILEOPS_RUNNER_IMAGE is required}
artifact_dir="$repo_dir/artifacts/preflight/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$artifact_dir"

"$repo_dir/scripts/build_tilefoundry_wheel.sh" | tee "$artifact_dir/tilefoundry-wheel.txt"
docker pull "$image" | tee "$artifact_dir/docker-pull.log"
docker image inspect "$image" --format '{{json .RepoDigests}} {{.Id}}' \
    | tee "$artifact_dir/image.txt"

"$repo_dir/scripts/tileops-container.sh" start | tee "$artifact_dir/container.txt"
"$repo_dir/scripts/tileops-container.sh" python scripts/ci/verify_runtime_stack.py \
    | tee "$artifact_dir/runtime-stack.log"
"$repo_dir/scripts/tileops-container.sh" python scripts/ci/verify_runner_image.py \
    | tee "$artifact_dir/runner-image.log"
"$repo_dir/scripts/tileops-container.sh" python /workspace/tileops-foundry-loop/scripts/preflight_gpu.py \
    | tee "$artifact_dir/gpu.json"
"$repo_dir/scripts/tileops-container.sh" python -c \
    'import importlib.metadata as m, json, pathlib, tilefoundry; print(json.dumps({"version": m.version("tilefoundry"), "module": str(pathlib.Path(tilefoundry.__file__).resolve())}))' \
    | tee "$artifact_dir/tilefoundry-installed.json"
"$repo_dir/scripts/tileops-container.sh" bash -lc \
    'python -m pip check > /tmp/tilefoundry-loop-pip-check.log 2>&1 || true; cat /tmp/tilefoundry-loop-pip-check.log' \
    | tee "$artifact_dir/pip-check.log"
"$repo_dir/scripts/tileops-container.sh" python \
    /workspace/tileops-foundry-loop/scripts/check_tilefoundry_environment.py \
    | tee "$artifact_dir/tilefoundry-dependency-closure.json"
"$repo_dir/scripts/tileops-container.sh" tilefoundry --help \
    | tee "$artifact_dir/tilefoundry-help.txt" >/dev/null

echo "$artifact_dir"
