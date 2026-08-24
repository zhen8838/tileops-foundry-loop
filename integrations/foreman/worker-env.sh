#!/usr/bin/env bash

if [[ -z ${TILEOPS_FOUNDRY_LOOP_ROOT:-} || -z ${FOREMAN_WORKTREE:-} || \
    -z ${FOREMAN_TASK:-} ]]; then
    echo "worker-env requires TILEOPS_FOUNDRY_LOOP_ROOT, FOREMAN_WORKTREE and FOREMAN_TASK" >&2
    return 1
fi

source "$TILEOPS_FOUNDRY_LOOP_ROOT/config/defaults.env"
if [[ -f "$TILEOPS_FOUNDRY_LOOP_ROOT/.env" ]]; then
    source "$TILEOPS_FOUNDRY_LOOP_ROOT/.env"
fi
cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
admission="$cache_root/worker-admissions/$FOREMAN_TASK.env"
[[ -f "$admission" ]] || { echo "missing worker admission: $admission" >&2; return 1; }
source "$admission"

# A restart may happen after Docker was stopped. Starting through the same helper
# refreshes the port and SSH target before Pi is relaunched in this pane.
cd "$FOREMAN_WORKTREE" || return
"$TILEOPS_FOUNDRY_LOOP_ROOT/scripts/tileops-container.sh" start >/dev/null || return
source "$admission"

export FOREMAN_WORKTREE TILEOPS_ROUND_HOST TILEOPS_ROUND_SLUG TILEOPS_TASK
export TILEOPS_WORKER_CONTAINER TILEOPS_SSH_IDENTITY TILEOPS_SSH_TARGET
export TILEOPS_PI_SSH_TARGET TILEOPS_PI_SSH_EXTENSION TILEOPS_AGENT_IMAGE
case ":$PATH:" in
    *":$TILEOPS_FOUNDRY_LOOP_ROOT/scripts:"*) ;;
    *) export PATH="$TILEOPS_FOUNDRY_LOOP_ROOT/scripts:$PATH" ;;
esac

cd "$TILEOPS_ROUND_HOST" || return
