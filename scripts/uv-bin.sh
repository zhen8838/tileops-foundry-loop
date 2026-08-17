# Sourced, never executed: sets `uv_bin` to the uv this machine installs.
#
# A foreman hook runs without a login shell's PATH, so finding uv cannot depend
# on the caller having one. Look where it is actually installed before giving up.

uv_bin=${TILEFOUNDRY_UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
    for candidate in "$HOME/bin/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [[ -x "$candidate" ]]; then
            uv_bin=$candidate
            break
        fi
    done
fi
if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
    echo "uv not found; set TILEFOUNDRY_UV_BIN or put uv on PATH" >&2
    exit 1
fi
