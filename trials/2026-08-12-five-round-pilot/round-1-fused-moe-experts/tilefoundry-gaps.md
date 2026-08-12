# TileFoundry gaps: fused routed experts

TileFoundry checkout: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.

## Vector runtime `IndexSelect` cannot lower to TIR

- Category: `lowering/codegen-blocker`
- Classification: `enhancement` (the current checkout has an explicit
  fail-closed test for this boundary; this is not a newly discovered semantic
  bug)
- Likely owner: `src/tilefoundry/passes/transforms/hir_to_tir.py`,
  `_lower_index_select`, plus the HIR-to-TIR `IndexSelect` specification.

Minimal reproducer: `index_select_lowering_repro.py`.

```bash
$HOST_HOME/TileFoundry/.venv/bin/python \
  index_select_lowering_repro.py
```

Expected: lower a rank-1 runtime route vector and produce a tensor whose first
dimension is the route count, matching evaluator/type-inference behavior.

Actual (`index-select-lowering-error.log`):

```text
NotImplementedError: IndexSelect HIR-to-TIR lowering supports only a one-element index
```

The full authored graph uses two `IndexSelect` operations with an index shape
of `[T*K]`: one for `[E,2F,H]` gate/up weights and one for `[E,H,F]` down
weights. Consequently all four manifest workloads are affected (`T*K` is
4,096 or 32,768). `tilefoundry analyze` and the evaluator accept the graph,
and `tilefoundry schedule` returns a small-shape CTA plan, but the plan cannot
pass this lowering boundary.

The handwritten runtime-indexed TileLang workaround in
`first_candidate_direct.py` preserves the value graph but performs no expert
grouping. On `qwen3-235b-decode` it measured 953.0823 ms versus 3.0369 ms for
the final grouped TileOPs candidate, a 313.84x latency cost. A useful lowering
needs either vector gather code generation that can be fused with consumers or
a route-aware grouped execution representation; materializing selected
production weights is not a viable workaround.

## Non-TileFoundry environment limitations

These are recorded for reproducibility and are not classified as TileFoundry
bugs:

- The persistent TileOPs image lacks `cupti-python`; the repository benchmark
  correctly failed closed until the explicitly reported CUDA-events fallback
  was enabled. Every final comparison uses that one timing method.
- H200 hardware counters are disabled by the host policy. NCU reports
  `ERR_NVGPUCTRPERM` (`evidence/ncu-permission.log`), so occupancy, DRAM and
  tensor-pipe percentages could not be collected. PyTorch CUDA traces provide
  launch count and per-kernel time instead.
- Host-side H200 evaluator execution could not create a cuBLAS handle while the
  persistent benchmark container owned the GPU. The identical immutable HIR
  module has a CPU-target alias solely for `check`; H200 `analyze` and
  `schedule` continue to use the H200 target.
