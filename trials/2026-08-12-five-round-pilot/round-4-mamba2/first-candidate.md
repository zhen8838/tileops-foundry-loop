# Round 4 blind first candidate

The blind phase ended at `2026-08-12T04:36:40.176182836+08:00`. This file and
every cited artifact were written before reading any incumbent body under
`src/tileops/kernels/mamba/**`, or the bodies of `CBProducerOp`,
`DaCumsumFwdOp`, `SSDChunkStateFwdOp`, `SSDStatePassingFwdOp`, and
`SSDChunkScanFwdOp`.

## Draft corrections established before implementation

- `load_manifest()` returns the merged op dictionary directly, not an
  `{"ops": ...}` wrapper. The structural read therefore uses
  `load_manifest()["Mamba2FwdOp"]`.
- The manifest label `mamba2-2p7b-b1-s2k` fixes `H=80`, while
  `workloads/mamba2_e2e.py` labels `H=80` as 1.3B and 2.7B as `H=128`. The
  manifest's exact values and labels are authoritative for this round.
- The existing workload always generates `dt_bias`, the benchmark passes it,
  and the existing test calls the default `return_final_states=False`. The
  current test reference returns only `y`. None of these are valid evidence
  for the primary no-bias, fixed-two-output contract.
- The current TileFoundry `Clamp` schema requires both `min_val` and
  `max_val`, and reduction is spelled `tf.reduce(..., kind="sum")`; the brief's
  pseudocode was adapted to those real DSL surfaces without changing math.

## Authored HIR

- `authored_hir.py:Mamba2Step.mamba2_step` is the typed one-token value graph.
  It uses FP16 model inputs, FP32 state/delta/decay/update/readout, static
  `G=1` head expansion, post-update output, and returns FP32 `y_t` plus FP32
  `next_state`.
- `authored-hir-check.json` passes `nan_inf` on both outputs and reports
  `f32[1,2,16]` and `f32[1,2,16,16]`.
- `authored-hir-analyze.json` completes compute-cost, memory, and H200 roofline
  analysis. Its fixed step totals are 2,696 FP32 flops, 11,064 bytes read, and
  8,992 bytes written. These are HIR analysis figures, not measurements.
- `authored-hir-schedule.json` records a 34-operation H200 CTA plan with
  `FEASIBLE_NOT_PROVEN`, objective 149 ns, and one search worker/first-plan.
- `state_scan_hir.py` and `state-scan-check.json` prove a four-iteration FP32
  ordered carry executes and returns a finite `f32[1,2,16,16]` state.
- `ordered_scan_repro.py` is the typed full sequence attempt. Dynamic token
  slicing fails with `unsupported ShapeDim Var`. The smaller
  `dynamic_scan_output_repro.py` fails at dynamic output insertion with
  `name 't' resolved to non-Expr Python value RangeSlice`, matching round 3.
- `hir_roundtrip.py` exercises canonical source emission and re-import. The
  emitted `authored_hir_roundtrip.py` contains bare `max_val=inf`, and re-import
  fails with `undefined name 'inf'`. The original exact-HIR check/analyze paths
  remain successful.

Authored HIR SHA256 is
`64330b914b5e63915f09d12b49c52df93885724c7ff77e1d650a12a7e09a4737`.

## Graph-traceable runtime twin

Sources are `evidence-sources/mamba2_blind_candidate.py` and
`evidence-sources/mamba2_blind_candidate.generated.cu` (generated CUDA SHA256
`bfcadbf8d24bd26b07a63e224241fa499d0c63225d3a7a4d85ba3e975b848422`).

The handwritten TileLang runtime maps one `(batch, head)` stream to one CTA,
holds one `[P,N]` FP32 state in 32 KiB shared memory, processes tokens in
strict order, writes each FP32 readout immediately, and writes the final FP32
state after the loop. It does not materialize `[B,S,H,P,N]`, dispatch to
`mamba_ssm`, or call any incumbent path. The generated source shows one
`mamba2_serial_kernel`, 128 threads, 32,792 bytes of dynamic shared storage,
the serial token loop, direct recurrence, and both output stores.

The first attempted runner used postponed Python annotations, causing the
nested TileLang tensor dtype closure to fail with `NameError: dtype`. Removing
that Python-only annotation mode resolved it; this is not a TileFoundry gap.
Only one runtime configuration was evaluated: 128 threads, direct recurrence,
no tuning. Compile/correctness/measurement wall time was under two minutes,
well below 0.02 H200 GPU-hours.

## Correctness

The exact command was:

```bash
$HOST_HOME/foreman/local/tileops-container.sh env \
  PYTHONPATH=$CONTAINER_WORKSPACE/tileops python scripts/mamba2_blind_candidate.py \
  --bench-trials 3 \
  --source-out $CONTAINER_WORKSPACE/tileops/mamba2_blind_candidate.generated.cu \
  --json-out $CONTAINER_WORKSPACE/tileops/blind-first-candidate.json \
  --junit-out $CONTAINER_WORKSPACE/tileops/blind-first-candidate.xml
```

The complete stdout/stderr is `blind-first-candidate.log`; structured results
and a two-case, zero-failure JUnit are
`artifacts/blind-first-candidate.json` and
`artifacts/blind-first-candidate.xml`.

| Fixture | Comparator | Output | max abs | mean abs | tolerance |
| --- | --- | --- | ---: | ---: | --- |
| S512/H4 FP16, 2 chunks | direct FP32 recurrence | y | 4.47035e-08 | 2.25549e-09 | 1e-2 / 1e-3 |
| S512/H4 FP16, 2 chunks | direct FP32 recurrence | final state | 2.23517e-08 | 1.02261e-09 | 1e-2 / 1e-3 |
| S512/H4 FP16, 2 chunks | official contract | y | 5.68107e-05 | 2.53421e-06 | 1e-2 / 1e-3 |
| S512/H4 FP16, 2 chunks | official contract | final state | 2.66731e-05 | 1.83333e-06 | 1e-2 / 1e-3 |
| S2048/H80 BF16 manifest | official contract | y | 0.00145105 | 2.69961e-05 | 2e-2 / 1e-3 |
| S2048/H80 BF16 manifest | official contract | final state | 0.000617206 | 1.71683e-05 | 2e-2 / 1e-3 |

Both fixtures return exact contract shapes and FP32 dtypes for both outputs.
The independent oracle is deliberately used on the smaller multi-chunk case;
the complete manifest row is checked against the separately installed official
implementation with `dt_bias=None`, `initial_states=None`,
`dt_softplus=True`, and `return_final_states=True`.

## Native CUPTI latency

Compilation, official execution, and correctness complete before timing. Each
trial retains both outputs, uses the repository's L2-reset/shifted-input
`bench_kernel`, and reports `metadata.timing="cupti"`.

| Trial | Samples | median ms | p10 ms | p90 ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 11.194595 | 11.181107 | 11.204467 |
| 1 | 10 | 11.197924 | 11.180882 | 11.211731 |
| 2 | 10 | 11.209492 | 11.178739 | 11.229139 |

The median of trial medians is 11.197924 ms. Raw samples are not summarized
away; all 30 values are in `artifacts/blind-first-candidate.json`.

## Limitations

- This correctness-first runtime has only 80 CTAs on the complete row and
  serializes all 2,048 tokens. It is about an order of magnitude slower than
  the preflight official/incumbent observations and is not a production path.
- It was required only to establish independent derivation and the first BF16
  row. The second FP16 manifest row and variants remain for the post-blind
  optimization/validation phase.
- The runtime twin is handwritten because the full HIR cannot bind the dynamic
  token index and output insertion. No claim is made that ordinary pipeline
  fusion or scan optimization is a TileFoundry bug.

## Blind source-read audit

All times are Asia/Shanghai. The source phase began after the brief/rule read
at approximately `2026-08-12T04:20+08:00`; the exact end is recorded above.
The following are every source surface read before sealing, in order. Parallel
batches are grouped on one line.

```text
brief.md; CLAUDE.md; .claude/domain-rules/{testing-budget,ops-design,benchmark}.md; docs/design/trust-model.md
src/tileops/ops/mamba2_fwd.py (public signature, validation, output and dispatch file only)
workloads/mamba2_e2e.py; tests/ops/test_mamba.py; benchmarks/ops/bench_mamba2_e2e.py; benchmarks/benchmark_base.py
tileops.manifest.load_manifest()["Mamba2FwdOp"] through the pinned container
installed mamba_ssm mamba_chunk_scan_combined signature and public wrapper source
round-4 preflight.log and preflight_smoke.py
TileFoundry README, tutorial/{migrate,optimize}.md, spec/{hir,parser,evaluator}.md
TileFoundry parser/evaluator/op/inspection tests for functions, loops, softplus, clamp, repeat_interleave, insert_slice and canonical printing
TileFoundry logical/runtime twin fixtures and worked-example file listings
round-2 and round-3 first-candidate/report/tilefoundry-gaps plus their authored HIR and minimal scan reproducers
unrelated TileLang convention reads: gla_recurrence.py, gated_deltanet_recurrence.py, elementwise.py, and targeted rg results under src/tileops/kernels with explicit mamba exclusions
benchmark_base.py CUPTI bench_kernel implementation and targeted call-site searches
```

No blind-phase command opened, searched inside, printed, source-inspected,
disassembled, or imported source from any forbidden incumbent implementation.
