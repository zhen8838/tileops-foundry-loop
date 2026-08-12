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
patch target; do not change the contract, benchmark, manifest, workload, wrapper,
or dispatch.
