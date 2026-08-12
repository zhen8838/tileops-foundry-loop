# Round 3 blind first candidate

Blind phase ended at `2026-08-12T03:13:27.898037498+08:00`.  This file and the
evidence it cites were written before reading
`src/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py` or any cached or
generated copy of that incumbent body.

## Authored HIR

- Executable one-step source: `authored_hir.py:GatedDeltaNetStep.delta_step`.
- Typed contract: FP16 inputs for q/k/v/g/beta, FP32 state carry, FP32 output
  and next state.  It computes `alpha=exp(g)`, `old=k^T state`,
  `delta=beta*(v-alpha*old)`, the post-update state, then `q^T next_state` with
  score scale 1.0.
- `authored-hir-check.json`: `tilefoundry check` passed `nan_inf` for both
  outputs.  Reported output types are `f32[1,2,16]` and
  `f32[1,2,16,16]`.
- `authored-hir-analyze.json`: `analyze --compute-cost --memory --roofline`
  completed against the installed `nvidia.h200_sxm` target.  It reports 3,782
  FP32 flops for the fixed one-step fixture and an 8,392-byte peak HBM value
  footprint.  These are HIR analysis figures, not runtime measurements.
- `state_scan_hir.py` and `state-scan-check.json`: a stable four-iteration
  recurrence carry proves `range`/`GridRegionExpr` loop-phi evaluation itself
  runs and returns a finite `[1,2,16,16]` FP16 final state.
- Full typed prefill attempt: `ordered_scan_repro.py:GatedDeltaNetPrefill`.
  This carries an FP32 state and FP32 output buffer, reads BTHD token windows,
  emits post-update outputs, then casts both contract outputs to FP16.

Current HIR limitation is narrower than “ordered scan is absent.”  Ordered
carry exists.  Full prefill cannot currently write each readout at the loop
induction position:

- `dynamic_scan_output_repro.py` is the smallest reproducer.
- `dynamic-scan-output-repro.log` reports
  `name 't' resolved to non-Expr Python value RangeSlice` when the loop's
  `tile(S, 1)` value is used as an `InsertSlice` offset.
- The scalar-`range` spelling for a multi-axis token read reports
  `unsupported indexer Name(id='t', ctx=Load())`; an earlier full scan attempt
  reaching dynamic insert reported `unsupported ShapeDim Var`.
- `ordered-scan-repro.log` preserves the full-contract failure.

Thus the authored step and carry are validated, while the full scan requires a
handwritten TileLang runtime twin.  No claim is made that ordinary scan
optimization is a TileFoundry bug.

## Runtime twin

Source:
`src/tileops/kernels/gated_deltanet/gated_deltanet_prefill_authored.py`.
Evidence runner: `scripts/gdn_prefill_blind_candidate.py`.

The runtime maps one `(B,H)` stream to one CTA, keeps only a 64 KiB
`[128,128]` FP32 state plus O(D) vectors in shared memory, processes tokens in
strict order, and casts `o` and `final_state` to the input dtype.  It creates no
per-token state history and no dense `[S,DK,DV]` temporary.  Its initial scope
is deliberately only the production specialization `B=1`, BTHD,
`DK=DV=128`, chunk 64, FP16/BF16.  It was injected through the existing
`kernel_map`; no incumbent path was called.

## Correctness

Small fixture command:

```bash
$HOST_HOME/foreman/local/tileops-container.sh env \
  PYTHONPATH=$CONTAINER_WORKSPACE/tileops python scripts/gdn_prefill_blind_candidate.py \
  --seq-len 64 --heads 2 --dtype float16
```

Artifact `blind-small-fp16.log`: candidate class was
`GatedDeltaNetPrefillAuthoredKernel`; `o` max abs error was
`0.0001983642578125`, final-state max abs error was
`0.000213623046875`, both within unchanged FP16 `atol=rtol=0.05`.

Production command:

```bash
$HOST_HOME/foreman/local/tileops-container.sh env \
  PYTHONPATH=$CONTAINER_WORKSPACE/tileops python scripts/gdn_prefill_blind_candidate.py \
  --seq-len 4096 --heads 16 --dtype bfloat16
```

Artifact `blind-production-4k-h16-bf16.log`: `o` max abs error was
`0.0048828125`, final-state max abs error was `0.004638671875`, both
within unchanged BF16 `atol=rtol=0.1`.  Shapes were
`o=[1,4096,16,128]`, `final_state=[1,16,128,128]`; both were BF16.

## Native CUPTI latency

Command added `--bench-trials 3` to the production command above.  Artifact
`blind-production-4k-h16-bf16-cupti.log` preserves every raw sample and reports
`metadata={"timing":"cupti"}` for all trials.  Compilation, reference
calculation, and correctness checks completed before timing.  The callable
returned and retained both outputs.

| Trial | Samples | Median ms | p10 ms | p90 ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 19.398247 | 19.090022 | 19.560328 |
| 1 | 10 | 17.278528 | 15.464344 | 19.704425 |
| 2 | 10 | 20.657372 | 18.967334 | 21.557039 |

The large between-trial spread is a known limitation of this first serial
candidate and will be retained in later comparisons rather than hidden.

## Known limitations and observed errors

- The runtime is a correctness-first serial state scan with only 16 CTAs on the
  production 4k row.  It is not yet a plausible SOTA candidate.
- It has no tuning surface and was tested at 128 threads only (one
  configuration, under one minute of compile/test wall time).
- It supports only the production specialization.  Public fallback behavior
  has not been changed.
- The first HIR attempt used Python `tuple[...]` return annotations and failed
  with `annotation did not resolve to TensorType, got GenericAlias`; current
  authored examples infer tuple returns, so the validated source follows that
  syntax.
- The first scan attempt omitted full-rank subscripts and failed before the
  intended test; that was corrected and is not treated as a capability gap.
- `state_scan_hir.py` initially overflowed with unconstrained independent
  random tensors.  Scaling its proof inputs made the carry-only capability
  test finite; production semantic correctness is established separately by
  the TileOPs oracle checks above.

## Blind source-read audit

All timestamps are Asia/Shanghai (`+08:00`).  Parallel commands below share
the recorded batch start timestamp.  The initial reads of `brief.md`,
`CLAUDE.md`, and Markdown file names were process/rule discovery, not source
reads; the first source batch began at `03:00:09`.

```text
2026-08-12T03:00:09.919511259  sed docs/design/trust-model.md; sed docs/design/ops-design.md
2026-08-12T03:00:09.919511259  sed docs/design/testing.md; sed benchmarks/benchmark_base.py
2026-08-12T03:00:09.919511259  yaml.safe_load linear_attention.yaml entry; sed workloads/linear_attention.py
2026-08-12T03:00:09.919511259  sed src/tileops/ops/gated_deltanet.py; sed tests/ops/test_gated_deltanet_prefill.py; sed benchmarks/ops/bench_gated_deltanet_prefill.py
2026-08-12T03:00:32.505620044  nl src/tileops/ops/gated_deltanet.py
2026-08-12T03:00:32.517661871  rg/nl workloads/linear_attention.py
2026-08-12T03:00:32.505620044  nl tests/ops/test_gated_deltanet_prefill.py; nl benchmarks/ops/bench_gated_deltanet_prefill.py
2026-08-12T03:00:55.361487347  nl workloads/linear_attention.py ranges 120:240 and 280:445
2026-08-12T03:00:55.372353059  nl src/tileops/ops/gated_deltanet.py range 283:445
2026-08-12T03:00:55.393913727  nl benchmarks/ops/bench_gated_deltanet_prefill.py
2026-08-12T03:01:44.212027932  rg delta_step and nl qwen3_5_35b_a3b/model.py
2026-08-12T03:02:06.645408700  nl qwen3_5_35b_a3b/model.py range 115:190
2026-08-12T03:02:06.671621543  ls reduction; nl reduction/cumsum.py (missing path)
2026-08-12T03:02:38.193723803  nl qwen3_5_35b_a3b/model.py range 1:55
2026-08-12T03:02:38.217506558  nl reduction/cumulative.py; nl reduction/_primitives.py
2026-08-12T03:02:38.206362992  nl src/tileops/kernels/kernel_base.py
2026-08-12T03:03:59.340515877  rg authored tuple annotations in TileFoundry tests/examples
2026-08-12T03:04:18.158048966  rg HIR range loops in TileFoundry tests/models
2026-08-12T03:04:30.971557482  nl TileFoundry test_parse_grid_region.py
2026-08-12T03:05:31.049131073  rg insert_slice in TileFoundry tests/models
2026-08-12T03:05:31.065156470  rg CudaTarget in qwen3_5_35b_a3b/model.py
2026-08-12T03:05:48.642843964  nl TileFoundry test_dsl_parse.py
2026-08-12T03:07:11.618512921  rg/nl reduction/cumulative.py
2026-08-12T03:07:11.632558261  rg shared-memory/synchronization conventions in reduction kernels
2026-08-12T03:07:21.923119020  nl reduction/cumulative.py and reduction/argreduce.py
2026-08-12T03:11:08.281064890  rg CUPTI/bench_kernel/profile_run in benchmarks and src/tileops
2026-08-12T03:11:08.275842575  rg/nl benchmarks/benchmark_base.py
2026-08-12T03:11:19.628714803  nl benchmarks/benchmark_base.py ranges 500:670 and 700:790
```

No command opened, searched within, printed, source-inspected, disassembled,
or imported source from the incumbent prefill kernel body before this record.
