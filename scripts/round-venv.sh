# Sourced, never executed: names a round's host environment.
#
# The hook that builds it and the pane environment that activates it both go
# through this function, so neither can drift from the other.

round_venv_path() {
    local worktree=$1
    local cache_root=${TILEOPS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tileops-runner}
    local resolved
    resolved=$(cd -- "$worktree" && pwd -P) || return 1
    printf '%s/round-venvs/%s\n' "$cache_root" "$(basename -- "$resolved")"
}
