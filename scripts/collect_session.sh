#!/usr/bin/env bash
# Copy a round worker's Claude Code session into the round, with a manifest.
#
# The transcript is the only record of how the round was actually driven -- what
# was tried, in what order, and what the agent saw when it changed its mind.
# `report.md` states conclusions; this states the path to them, which is what a
# later experiment compares against. It lives outside the round by default, under
# a per-directory slug in `~/.claude/projects`, and is deleted with the tool's own
# history rather than with the round, so it is copied in before teardown.
#
# Session files carry raw tool output and are read-restricted at the source; the
# copies keep those permissions. Redaction for publication is `archive_trial.py`'s
# job, not this script's.
set -euo pipefail

round=${1:?usage: $0 ROUND_DIR}
round=$(cd -- "$round" && pwd -P)
projects=${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}

# Claude Code names a project directory after the working directory the session
# started in, with each character that is not a letter or a digit replaced by a
# dash -- dots in a username included. The separator-only form is tried too, in
# case that rule ever narrows.
source_dir=""
for slug in "${round//[^A-Za-z0-9]/-}" "${round//\//-}"; do
    if [[ -d "$projects/$slug" ]]; then
        source_dir="$projects/$slug"
        break
    fi
done
[[ -n "$source_dir" ]] || {
    echo "no session directory for $round under $projects" >&2
    exit 1
}

destination="$round/session"
mkdir -p "$destination"
shopt -s nullglob
transcripts=("$source_dir"/*.jsonl)
(( ${#transcripts[@]} > 0 )) || {
    echo "no transcript in $source_dir" >&2
    exit 1
}
for transcript in "${transcripts[@]}"; do
    cp -p -- "$transcript" "$destination/"
done
if [[ -d "$source_dir/memory" ]]; then
    mkdir -p "$destination/memory"
    cp -p -- "$source_dir/memory"/* "$destination/memory/" 2>/dev/null || true
fi

{
    printf '{\n'
    printf '  "round": %s,\n' "\"$(basename -- "$round")\""
    printf '  "source": %s,\n' "\"$source_dir\""
    printf '  "collected": %s,\n' "\"$(date -Iseconds)\""
    printf '  "transcripts": [\n'
    separator=""
    for transcript in "${transcripts[@]}"; do
        name=$(basename -- "$transcript")
        printf '%s    {"file": "%s", "bytes": %s, "lines": %s, "sha256": "%s"}' \
            "$separator" "$name" \
            "$(stat -c %s -- "$transcript")" \
            "$(wc -l <"$transcript")" \
            "$(sha256sum -- "$transcript" | awk '{print $1}')"
        separator=$',\n'
    done
    printf '\n  ]\n}\n'
} >"$destination/MANIFEST.json"

printf '%s\n' "$destination"
