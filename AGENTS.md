# Repository Instructions

This repository is intentionally small. The versioned round contract lives in
`templates/round/`; the shell scripts only create the round, build the admitted
TileFoundry wheel, start the SSH-enabled TileOPs runner, and connect Foreman/Pi.

Keep round-specific HIR, experiments, logs, profiles, and reports under the
generated `rounds/<slug>/` directory. Git commit, push, PR, and CI remain host
operations on the TileOPs worktree. Do not add a second policy document or a
historical trial archive here.
