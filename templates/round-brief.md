# {{OPERATOR}}

Run the worker loop and pass every gate in `PLAYBOOK.md`.

- Scope: `{{SCOPE}}`
- Strongest same-contract external baseline: {{BASELINE}}
- TileOPs base: `{{TILEOPS_BASE}}`
- TileFoundry wheel commit: `{{TILEFOUNDRY_COMMIT}}`
- Host round directory: `{{ROUND_DIR}}`
- Container round directory: `/workspace/tileops-loop-state/{{SLUG}}`

Make this operator SOTA on H200 with a new TileLang kernel. The public Op,
manifest workloads, correctness surface, and benchmark are the contract.
Work from this round directory. The TileOPs worktree is only the final kernel
and production-dispatch patch target. Shape-aware dispatch may select different
kernels or fusion boundaries while preserving the public Op signature and math.
The unchanged benchmark must reach it through normal Op construction, without a
candidate-only switch. Do not change the contract, benchmark, manifest, workload,
reference, or evaluation path.

This directory starts empty. There is no scaffold: author what the round needs,
and run `scripts/check_round.py` early — it names everything it refuses, and that
list is the contract. `tilefoundry tutorial|spec|models` teaches the workflow;
author your own HIR rather than copying a graph. `analyze` and `schedule` run in
this pane, `tileops-run tilefoundry check` in the container.

Before authoring, read `/workspace/tileops-foundry-loop/knowledge/tilelang.md`
(measured, tagged with the build — check before relying, and write verdicts under
`TileLang Notes` in the report) and `knowledge/tilefoundry.md` (surfaces known to
refuse a legal program — cite the id, keep the faithful program).

No command may exceed 10 minutes; the tool kills it and the sweep is lost. Record
the round's first compile and a later one of the same kernel, so the shared kernel
cache is measured. Ask the human only when a gate condition itself is in question.
