# TileFoundry gaps: round 4 Mamba2FwdOp

TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
The exact one-step graph is `authored_hir.py`; the full ordered recurrence is
`ordered_scan_repro.py`.

## TF-R4-M2-01: dynamic loop token indexing and output insertion

- Classification: `semantic-blocker`; status: `duplicate` of round 3
  `TF-R3-GDN-01`, with Mamba-2 as an additional affected graph.
- Smallest reproducer: `dynamic_scan_output_repro.py`. It creates an `[4,4]`
  output, iterates `for t in tile(4, 1)`, and inserts `x[t, :]` at `(t, 0)`.
- Expected: bind the ordered-loop induction position as a tensor index and
  `InsertSlice` offset, yielding a copy of `x`.
- Actual: `dynamic-scan-output-repro.log` reports
  `name 't' resolved to non-Expr Python value RangeSlice`. The full typed
  Mamba-2 reproducer fails one step earlier with `unsupported ShapeDim Var`
  while forming a dynamic token slice; see `ordered-scan-repro.log`.
- Affected rows: both primary rows and every public Mamba-2 sequence shape.
  The recurrence must dynamically read each token and write every post-update
  readout; final state alone does not satisfy the two-output contract.
- Workaround and measured cost: the handwritten TileLang twin in
  `evidence-sources/mamba2_blind_candidate.py` serially scans tokens without a
  dense state history. Its S2048/H80 BF16 native-CUPTI median-of-three is
  11.197924 ms. This is roughly 8.6x the preflight official observation and
  16.2x the preflight incumbent observation; final same-process comparisons
  will replace those approximate ratios in `report.md`.
- Likely owner: HIR grid-region induction binding, tensor subscript parsing,
  and dynamic `InsertSlice` offset typing. This evidence does not prescribe
  which layer should change.

This is not round 2's symbolic `RepeatInterleave` gap. Both Mamba-2 primary
rows have static `G=1`, and the authored step's static
`repeat_interleave(..., repeats=HEADS)` passes check, analyze, and schedule.
It also does not claim ordered carry is absent: `state-scan-check.json` proves
four FP32 carry iterations execute.

## TF-R4-M2-02: canonical printer emits unbound positive infinity

- Classification: `ergonomics`; status: `new`.
- Smallest reproducer: run `hir_roundtrip.py`. It imports the valid authored
  step, calls `tilefoundry.inspection.as_script`, writes
  `authored_hir_roundtrip.py`, then imports the emitted source.
- Expected: canonical source for `Clamp(max_val=+inf)` is self-contained and
  re-imports to an equivalent module.
- Actual: the printer emits `clamp(..., max_val=inf)` without defining or
  importing `inf`; re-import fails with `undefined name 'inf'`. The complete
  traceback and emitted source are in `hir-roundtrip.log` and
  `authored_hir_roundtrip.py`. The original source independently passes check,
  analyze, and schedule, proving this is the source-to-source surface.
- Affected rows: no runtime row is numerically blocked, but every exact
  primary-contract description uses the specified `dt_limit=(0,+inf)` and
  cannot round-trip through canonical source. Replacing infinity with a finite
  constant changes edge-case semantics, so it is not used as the exact graph.
- Workaround and measured cost: retain the authored source instead of the
  canonical emission; zero measured runtime cost, but canonical source cannot
  serve as a regenerable artifact.
- Likely owner: the HIR Python printer's scalar attribute rendering and/or the
  parser's static namespace for floating-point infinity.

## Non-gap environment limit

NCU counters are unavailable under host policy (`ERR_NVGPUCTRPERM`). This is
not a TileFoundry defect; final evidence uses native CUPTI, PyTorch traces,
generated source/resource inspection, and controlled ablations.
