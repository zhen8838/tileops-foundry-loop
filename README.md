# TileOPs Foundry Loop

Reusable infrastructure for turning a TileOPs operator contract into a
TileFoundry-derived TileLang kernel and a measured TileOPs performance PR.

[PLAYBOOK.md](PLAYBOOK.md) is the single workflow contract. Concrete plans and
fillable templates link to it instead of copying policy.

## Repository layout

```text
PLAYBOOK.md             Normative workflow and agent prompt
plans/                  Concrete operator goals only
config/                 Shared defaults and machine environment example
scripts/                Container, preflight, dispatch, and PR commands
templates/              Fillable round records and PR input
examples/               Rendered PR reference
tileops_foundry_loop/   PR schema, validation, and renderer
tests/                  Contract tests
```

## Start Here

```bash
cp config/local.env.example .env
# Edit machine-local paths, then read PLAYBOOK.md completely.
```

The reusable five-round goal is [plans/five-round.md](plans/five-round.md).
The generated public result is visible in
[examples/fused-moe/pr-body.md](examples/fused-moe/pr-body.md). Run
`make check test` after repository changes.

## Local status

This checkout is intentionally local-only until its GitHub owner, repository
name, and visibility are chosen. It can be reviewed and amended here without
creating external state.
