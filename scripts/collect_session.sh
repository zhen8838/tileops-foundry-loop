#!/usr/bin/env bash
# Copy a round worker's Claude Code session into the round, with a manifest.
#
# `report.md` states the conclusions; the transcript states the path to them, and
# it lives outside the round -- under a per-directory slug in the agent tool's own
# history -- so teardown can lose it. Copies keep the source's permissions;
# redaction for publication is `archive_trial.py`'s job.
set -euo pipefail

round=${1:?usage: $0 ROUND_DIR}
round=$(cd -- "$round" && pwd -P)
projects=${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}

# The slug is the directory the session started in, with every character that is
# not a letter or a digit replaced by a dash -- dots in a username included.
source_dir=""
for slug in "${round//[^A-Za-z0-9]/-}" "${round//\//-}"; do
    [[ -d "$projects/$slug" ]] && source_dir="$projects/$slug" && break
done
[[ -n "$source_dir" ]] || {
    echo "no session directory for $round under $projects" >&2
    exit 1
}

destination="$round/session"
mkdir -p "$destination"
shopt -s nullglob
transcripts=("$source_dir"/*.jsonl)
(( ${#transcripts[@]} > 0 )) || { echo "no transcript in $source_dir" >&2; exit 1; }
cp -p -- "${transcripts[@]}" "$destination/"
if [[ -d "$source_dir/memory" ]]; then
    mkdir -p "$destination/memory"
    cp -p -- "$source_dir/memory"/* "$destination/memory/" 2>/dev/null || true
fi

{
    printf '{\n  "round": "%s",\n  "source": "%s",\n  "collected": "%s",\n  "transcripts": [\n' \
        "$(basename -- "$round")" "$source_dir" "$(date -Iseconds)"
    separator=""
    for transcript in "${transcripts[@]}"; do
        printf '%s    {"file": "%s", "bytes": %s, "lines": %s, "sha256": "%s"}' \
            "$separator" "$(basename -- "$transcript")" \
            "$(stat -c %s -- "$transcript")" "$(wc -l <"$transcript")" \
            "$(sha256sum -- "$transcript" | awk '{print $1}')"
        separator=$',\n'
    done
    printf '\n  ]\n}\n'
} >"$destination/MANIFEST.json"

printf '%s\n' "$destination"
