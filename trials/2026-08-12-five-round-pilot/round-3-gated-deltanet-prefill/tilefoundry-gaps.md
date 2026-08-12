# TileFoundry gaps: round 3 Gated DeltaNet prefill

TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
The validated one-step graph is `authored_hir.py`; `state_scan_hir.py` proves
that an ordered FP32 loop carry is executable. The full typed prefill attempt
is `ordered_scan_repro.py`.

## TF-R3-GDN-01: a loop tile cannot index a dynamic output insertion

- Classification: `semantic-blocker`; status: `new`.
- Smallest reproducer: `dynamic_scan_output_repro.py`. It creates an
  `[4,4]` output, iterates `for t in tile(4, 1)`, and attempts
  `insert_slice(out, x[t, :], (t, 0))`.
- Expected: bind the loop induction position as a valid dynamic tensor index
  and `InsertSlice` offset, producing a copy of `x` after four ordered writes.
- Actual: `tilefoundry check` exits 1 with
  `name 't' resolved to non-Expr Python value RangeSlice`; complete output is
  in `dynamic-scan-output-repro.log`. The full-contract reproducer reaches the
  same boundary. A scalar `range` spelling instead fails the multi-axis token
  read with `unsupported indexer Name(id='t', ctx=Load())`; an earlier dynamic
  insertion spelling reported `unsupported ShapeDim Var`, preserved in
  `ordered-scan-repro.log`.
- Affected workloads: all 26 production manifest rows. Every row requires an
  ordered carry plus a post-update readout written at each BTHD sequence
  position. The final state alone is insufficient for the public two-output
  contract.
- Workaround and measured cost: the graph-traceable handwritten TileLang twin
  in `gated_deltanet_prefill_authored.py` maps one head to one CTA and keeps an
  FP32 `[128,128]` state in shared memory. On the 4k/H16 BF16 row its three
  native-CUPTI medians were 19.398247, 17.278528, and 20.657372 ms. It creates
  no dense per-token state history, but only launches 16 serial scan CTAs and
  is not a viable optimized production implementation. Its median-of-three
  19.398247 ms is 19.85x slower than the final 4k/H16 BF16 candidate's
  0.9772 ms; the optimized workaround uses the incumbent chunked/fused
  pipeline and is explicitly not relabeled as TileFoundry-derived.
- Likely owner: the DSL parser/binding representation for `GridRegionExpr`
  induction values, tensor index parsing, and the HIR `InsertSlice` dynamic
  offset type relation. This reproducer does not establish which of those
  layers should own a fix.

Run the minimal reproducer with:

```bash
$HOST_HOME/TileFoundry/.venv/bin/tilefoundry check \
  dynamic_scan_output_repro.py:ScanCopy.scan_copy \
  --target cpu.native --inputs random
```

This is not a duplicate of round 1's vector `IndexSelect` HIR-to-TIR lowering
boundary or round 2's UINT8 type, symbolic `RepeatInterleave`, and i32 schedule
facts gaps. It also is not a claim that ordered scan itself is absent:
`state-scan-check.json` records a successful four-iteration carry-only check.

## Non-gap environment limit

NCU performance counters remain unavailable under the host policy
(`ERR_NVGPUCTRPERM`). This is not a TileFoundry defect. Final performance
evidence therefore uses native CUPTI, PyTorch CUDA traces, generated CUDA
source, resource-usage metadata, and controlled partition ablations.
