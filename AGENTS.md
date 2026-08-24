# Repository Instructions

This repository is intentionally small. The versioned round contract lives in
`template/brief.md`; `setup` installs the Foreman hooks that create rounds, start the
SSH-enabled TileOPs runner, archive sessions, and release per-round resources.

Keep round-specific HIR, experiments, logs, profiles, and reports under the
generated `rounds/<slug>/` directory. Git and GitHub operations use the mounted
TileOPs gitdir and read-only host credentials; their policy still belongs to
TileOPs. Do not add a second policy document or a historical trial archive here.
