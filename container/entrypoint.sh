#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd /run/tileops
chmod 0755 /run/sshd /run/tileops
[[ -n ${TILEOPS_SSH_PUBLIC_KEY:-} ]] || {
    echo "TILEOPS_SSH_PUBLIC_KEY is required" >&2
    exit 2
}
printf '%s\n' "$TILEOPS_SSH_PUBLIC_KEY" > /run/tileops/authorized_keys
chmod 0600 /run/tileops/authorized_keys

install -d -m 0700 /root/.ssh
if [[ -n ${TILEOPS_HOST_SSH_DIR:-} && -d $TILEOPS_HOST_SSH_DIR ]]; then
    for path in "$TILEOPS_HOST_SSH_DIR"/*; do
        [[ -f "$path" ]] || continue
        name=${path##*/}
        [[ "$name" == environment ]] || ln -sfn "$path" "/root/.ssh/$name"
    done
fi

git_config=/root/.gitconfig
: >"$git_config"
if [[ -f /opt/tileops-host-auth/gitconfig ]]; then
    git config --file "$git_config" include.path /opt/tileops-host-auth/gitconfig
fi
git config --file "$git_config" --add safe.directory /workspace/tileops
git config --file "$git_config" --add safe.directory /workspace/tilefoundry

if [[ -x /opt/claude-code/bin/claude.exe ]]; then
    ln -sfn /opt/claude-code/bin/claude.exe /usr/local/bin/claude
    # Answer the first-run menus here. Unanswered, they stop the round on a theme,
    # a login method, a folder-trust question and a bypass-mode confirmation.
    python3 - <<'SEED'
import json
import pathlib

config = pathlib.Path("/root/.claude.json")
state = json.loads(config.read_text()) if config.is_file() else {}
state.update({"hasCompletedOnboarding": True, "theme": "dark", "installMethod": "global"})
projects = state.setdefault("projects", {})
projects.setdefault("/workspace/round", {})["hasTrustDialogAccepted"] = True
config.write_text(json.dumps(state))

settings = pathlib.Path("/root/.claude/settings.json")
settings.parent.mkdir(parents=True, exist_ok=True)
chosen = json.loads(settings.read_text()) if settings.is_file() else {}
chosen.update(
    {
        "skipDangerousModePermissionPrompt": True,
        "switchModelsOnFlag": False,
        "theme": "dark",
    }
)
settings.write_text(json.dumps(chosen))
SEED
fi

if [[ -x /usr/local/bin/gh && -d ${GH_CONFIG_DIR:-} ]]; then
    for host in github.com gist.github.com; do
        git config --file "$git_config" --add "credential.https://$host.helper" ""
        git config --file "$git_config" --add "credential.https://$host.helper" \
            '!/usr/local/bin/gh auth git-credential'
    done
fi
chmod 0600 "$git_config"

# sshd constructs a minimal login environment instead of inheriting the
# container environment. Preserve the compiler, GPU and cache settings that
# define this round without exposing the SSH key itself.
environment_file=/root/.ssh/environment
: >"$environment_file"
for name in \
    PATH LD_LIBRARY_PATH LIBRARY_PATH CUDA_HOME CUDA_VERSION CUDA_VISIBLE_DEVICES \
    NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES NVCC_THREADS \
    GIT_CONFIG_GLOBAL GIT_OPTIONAL_LOCKS GH_CONFIG_DIR PYTHONUNBUFFERED \
    TILEOPS_HOST_SSH_DIR TILEOPS_PHYSICAL_GPU IS_SANDBOX \
    TILELANG_CACHE_DIR TILELANG_TMP_DIR TRITON_CACHE_DIR \
    HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
    http_proxy https_proxy all_proxy no_proxy; do
    if [[ -v $name ]]; then
        value=${!name}
        [[ "$value" != *$'\n'* ]] || {
            echo "$name contains a newline and cannot be exported to SSH" >&2
            exit 2
        }
        printf '%s=%s\n' "$name" "$value" >>"$environment_file"
    fi
done
chmod 0600 "$environment_file"

ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd -t
exec /usr/sbin/sshd -D -e
