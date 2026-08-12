# TileOPs five-round TileFoundry gap inventory

Inventory baseline: TileFoundry `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
Source records and minimal reproducers are retained under the five round
directories beside this file. Environment-only limits such as
`ERR_NVGPUCTRPERM` are excluded.

## Deduplicated gaps

| ID | Classification | Rounds | Status | Evidence and effect | Disposition |
| --- | --- | --- | --- | --- | --- |
| `TF-X-SCAN-01` | `semantic-blocker` | 3, 4 | **fixed by TileFoundry PR #90** | A two-argument `tile(extent, step)` bound a parser-side `RangeSlice`. `x[t, :]` left the induction `Var` in the inferred slice extent (`unsupported ShapeDim Var`), while `insert_slice(..., offsets=(t, 0))` rejected `t` as a non-`Expr` `RangeSlice`. Carry-only loops and runtime scalar `InsertSlice` offsets already worked. The Gated DeltaNet and Mamba-2 full-sequence descriptions both required the missing read/write window. | Repaired as one parser/HIR/TIR window-indexing contract. Both minimal reproducers and both complete authored HIR descriptions now pass. |
| `TF-R1-MOE-01` | `lowering/codegen-blocker` | 1 | enhancement | Vector runtime `IndexSelect` is accepted by the evaluator and analysis, but HIR-to-TIR only lowers a one-element index. The route-vector workaround was 313.84x slower on the recorded decode workload. | Deferred P1. Needs a route-aware lowering decision; materializing selected production weights is not acceptable. |
| `TF-R2-W4-01` | `semantic-blocker` | 2 | new | The dtype surface rejects `u8`, so packed W4 weights and zero points cannot retain their public storage contract. An i32 carrier is 4x larger. | Deferred P1. Treat as a dtype/storage tranche, independently of scan indexing. |
| `TF-R2-W4-02` | `semantic-blocker` | 2 | new | `RepeatInterleave` cannot infer `(K // 128) * 128` from a symbolic repeated extent. Six static specializations would be required. | Deferred P1. Fix in symbolic shape inference with its own dynamic-shape acceptance suite. |
| `TF-R2-W4-03` | `performance-blocker` | 2 | new | Scheduling an f16-output graph with i32 unpack intermediates fails because the H200 target has no dense i32 peak fact. | Deferred P2. First decide whether integer intermediates require a throughput fact or non-dominant-dtype scheduling semantics. |
| `TF-R4-M2-02` | `ergonomics` | 4 | new | Canonical source prints `Clamp(max_val=+inf)` as bare `inf`; re-import fails because that name is unbound. Runtime rows are unaffected. | Deferred P2. Small printer/parser repair after semantic blockers. |
| `TF-R5-FFT-01` | `semantic-blocker` | 5 | new | Direct `complex64` and `complex128` tensor dtypes are rejected. Paired f32 preserves c64 arithmetic but not the public complex tensor contract. | Deferred P1. Requires an end-to-end complex dtype design across type rules, evaluator, and lowering. |
| `TF-R5-FFT-02` | `semantic-blocker` | 5 | new | `f64` is rejected, so paired-real HIR cannot exactly describe complex128 even without a complex dtype. | Deferred P1. Related to, but not implied by, complex dtype support. |

## Selected coherent repair

`TF-X-SCAN-01` is the only duplicated blocker. The smallest shared program is:

```python
out = tf.zeros(shape=(4, 4), dtype="f32")
for t in tile(4, 2):
    out = tf.insert_slice(out, x[t, :], (t, 0))
return out
```

Its observable contract is two windows, `[0:2]` and `[2:4]`, copied to the
same coordinates. This catches both failures reported by rounds 3 and 4 and
also prevents a step from being applied twice. The repair must carry that
window coordinate through parser binding, slice shape inference, evaluation,
and HIR-to-TIR lowering. It must not broaden into vector gather, dtype support,
target throughput facts, or printer cleanup.

The finalized implementation plan is
`docs/plans/tileops-scan-window-indexing/PLAN.md`. After its PR, the two minimal
scan-copy reproducers and the full Gated DeltaNet/Mamba-2 authored descriptions
are rerun. The affected TileOPs candidate benchmarks are rerun before and after
as a regression check; this semantic repair is not expected to change their
handwritten TileLang latency.

## Repair result

- TileFoundry PR: https://github.com/tile-ai/TileFoundry/pull/90 at
  `e0c46ff111bfbb0cb64b21b6209efe6979b7a6a8`; CI passed and GitHub reports the
  PR mergeable.
- Validation: 919 source tests passed with one skip; the 52-test installed and
  source blast-radius suites passed; all four round-3/round-4 minimal and full
  authored reproducers passed independently after the repair.
- Round 3 post-repair benchmark: 26/26 manifest benchmark cases passed and the
  candidate remained faster than FLA on every row. Raw output:
  `post-repair/round-3-profile_run.log`.
- Round 4 post-repair benchmark: 2/2 primary manifest cases passed. Candidate
  latency was 0.8952/1.3559 ms versus official mamba_ssm 0.8196/1.2053 ms, so
  its classification remains no improvement. Raw output:
  `post-repair/round-4-profile_run.log`.
- No second blocker appeared when the correct complete authored selectors were
  used. The seven deferred gaps above remain independently reproducible and
  were deliberately excluded from the coherent scan-window repair.
