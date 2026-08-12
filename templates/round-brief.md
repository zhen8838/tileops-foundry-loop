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
