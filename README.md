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
integrations/           Thin Foreman hook and prompt adapters
templates/              Fillable round records and PR input
examples/               Rendered PR reference
trials/                 Sanitized multi-round outputs and retrospectives
tileops_foundry_loop/   PR schema, validation, and renderer
tests/                  Contract tests
```

## Start Here

```bash
cp config/local.env.example .env
# Edit machine-local paths, then:
source .env
# Read PLAYBOOK.md completely.
```

The reusable five-round goal is [plans/five-round.md](plans/five-round.md).
The generated public result is visible in
[examples/fused-moe/pr-body.md](examples/fused-moe/pr-body.md). Run
`make check test` after repository changes.

The main agent prepares but does not monitor a round:

```bash
./scripts/preflight.sh
uv run python scripts/new_round.py --slug <slug> --scope <scope> \
  --operator <operator> --baseline '<same-contract baseline>' \
  --tileops-repo "$TILEOPS_REPO" --tilefoundry-repo "$TILEFOUNDRY_REPO" \
  --root "$TILEOPS_LOOP_STATE_ROOT"
./scripts/dispatch_round.sh <task> perf/<task> "$TILEOPS_LOOP_STATE_ROOT/<slug>/brief.md"
```

Foreman's project hook creates one persistent container per worker worktree.
The container installs the admitted TileFoundry wheel and never mounts the
TileFoundry source checkout.

Before Foreman starts each Agent Session, it sources the versioned worker
environment into that pane. It adds the repository's `tileops-run` wrapper to
that session's `PATH` and moves the session to its round directory. The wrapper
prefixes unchanged commands and maps that directory into the worker's persistent
container. The TileOPs worktree remains only as a kernel, shape-aware production
dispatch, and correctness-test patch target. The final gate rejects evaluation
contract changes and public Op signature changes. Nothing is installed into a
user bin directory or added to the container image.

The first pilot's raw outputs and human adjudication are archived in
[trials/2026-08-12-five-round-pilot/](trials/2026-08-12-five-round-pilot/README.md).

Import a completed local trial with:

```bash
uv run python scripts/archive_trial.py <state-directory> trials/<date-and-name>
```

The canonical public repository is
[zhen8838/tileops-foundry-loop](https://github.com/zhen8838/tileops-foundry-loop).
