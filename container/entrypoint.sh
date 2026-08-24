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
ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd -t
exec /usr/sbin/sshd -D -e
