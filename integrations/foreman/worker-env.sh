# Source this file in a Foreman pane before starting the Agent Session.

if [[ -z ${TILEOPS_FOUNDRY_LOOP_ROOT:-} || -z ${FOREMAN_WORKTREE:-} || -z ${FOREMAN_TASK:-} ]]; then
    echo "worker-env requires TILEOPS_FOUNDRY_LOOP_ROOT, FOREMAN_WORKTREE, and FOREMAN_TASK" >&2
    return 1
fi

# shellcheck disable=SC1091
source "$TILEOPS_FOUNDRY_LOOP_ROOT/config/defaults.env"
if [[ -f "$TILEOPS_FOUNDRY_LOOP_ROOT/.env" ]]; then
    # shellcheck disable=SC1091
    source "$TILEOPS_FOUNDRY_LOOP_ROOT/.env"
fi

tileops_cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
tileops_admission="$tileops_cache_root/worker-admissions/$FOREMAN_TASK.env"
if [[ ! -f "$tileops_admission" ]]; then
    echo "worker admission is missing: $tileops_admission" >&2
    return 1
fi
# shellcheck disable=SC1090
source "$tileops_admission"

if [[ -n ${TILEOPS_DOCKER_BOOTSTRAP:-} ]]; then
    "$TILEOPS_DOCKER_BOOTSTRAP" || return
fi
if [[ -n ${TILEOPS_DOCKER_HOST:-} ]]; then
    export DOCKER_HOST=$TILEOPS_DOCKER_HOST
fi

TILEOPS_WORKER_CONTAINER=$(
    cd "$FOREMAN_WORKTREE" &&
        "$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/tileops-container.sh" name
) || return
FOREMAN_WORKTREE=$(cd "$FOREMAN_WORKTREE" && pwd -P) || return
TILEOPS_ROUND_HOST=$(cd "$TILEOPS_ROUND_HOST" && pwd -P) || return
export FOREMAN_WORKTREE TILEOPS_WORKER_CONTAINER TILEOPS_ROUND_HOST TILEOPS_ROUND_SLUG

case ":$PATH:" in
    *":$TILEOPS_FOUNDRY_LOOP_ROOT/scripts:"*) ;;
    *) export PATH="$TILEOPS_FOUNDRY_LOOP_ROOT/scripts:$PATH" ;;
esac
