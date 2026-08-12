#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 GPU COMMAND [ARG...]" >&2
    exit 2
fi

gpu=$1
shift
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU must be a numeric index" >&2
    exit 2
fi

lock_root=${TILEOPS_FOUNDRY_LOOP_LOCK_ROOT:-/tmp/tileops-foundry-loop-locks}
mkdir -p "$lock_root"
exec 9>"$lock_root/gpu-$gpu.lock"
flock 9
exec "$@"
