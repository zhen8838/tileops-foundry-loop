# Round 4 brief: Mamba2FwdOp

## Objective

Produce a correct TileLang implementation derived through a TileFoundry authored-HIR workflow
for the primary `Mamba2FwdOp` contract, then optimize both primary manifest workloads on NVIDIA
H200. The strongest runnable same-math external baseline is the official `mamba_ssm` 2.3.1
Triton `mamba_chunk_scan_combined` implementation. Take an independently useful result through a
mergeable TileOPs PR; if no honest performance PR is justified, finish with reproducible evidence
and concrete blockers.

This is round 4 of the fixed five-round sequence. Do not use or substitute a GQA operator. You
are the only implementation agent for this round: do not dispatch another agent and do not edit
`$HOST_HOME/TileFoundry`.

## Fixed source and environment

- TileOPs base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`.
- TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
- Pinned image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`, digest
  `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`.
- Foreman creates one persistent container for this worktree. Run every TileOPs Python command,
  test, benchmark, and profiler through:

  ```bash
  $HOST_HOME/foreman/local/tileops-container.sh <command> [args...]
  ```

  The first call starts the container; later calls use `docker exec` in the same container on the
  same selected GPU. `shell` enters it interactively. Do not create another ad hoc container.
- Inside the container, `$CONTAINER_WORKSPACE/tileops` is writable, `$CONTAINER_WORKSPACE/tilefoundry` is read-only,
  and `$CI_CACHE` persists. The TileOPs common Git directory is mounted read-only and
  `GIT_OPTIONAL_LOCKS=0` is set.
- Use TileFoundry on the host only through
  `$HOST_HOME/TileFoundry/.venv/bin/<tool>`. Do not rebuild that environment.
- Do not install, upgrade, or replace packages. The image has CUDA 12.9, Python 3.12, PyTorch
  2.10.0+cu129, TileLang 0.1.11+cu129.gitafcebed1, `mamba_ssm` 2.3.1, and
  `cupti-python==12.8.0`.
- Native CUPTI timing is required for final latency. NCU performance counters are unavailable on
  this host with `ERR_NVGPUCTRPERM`; use CUPTI, PyTorch profiler, generated-source inspection,
  and controlled ablations after preserving the existing environment reproducer.
- Shared artifacts belong under
  `$TRIAL_SOURCE/round-4-mamba2/`.
- `preflight.log` in that directory records the structural manifest, installed baseline
  signature, native output dtypes, two-output correctness against the incumbent, and initial
  same-contract latency samples for both primary rows.

## Authoritative sources and blind boundary

Before producing the first correct candidate, read only these contract surfaces:

- Load `Mamba2FwdOp` structurally through `tileops.manifest`; the entry is in
  `src/tileops/manifest/mamba.yaml`.
- `src/tileops/ops/mamba2_fwd.py`: public constructor/forward signature, validation, output
  contract, and dispatch surface only. Do not follow its stage imports into incumbent bodies.
- `workloads/mamba2_e2e.py`: shapes and input distributions.
- `tests/ops/test_mamba.py`: `mamba2_fwd_ref`, public correctness fixtures, and tolerances.
- `benchmarks/ops/bench_mamba2_e2e.py` and `benchmarks/benchmark_base.py`.
- Installed `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` public wrapper and
  source as the external baseline.
- TileFoundry specs, tutorial, CLI, authored-HIR examples, and unrelated TileLang kernels needed
  only for repository conventions.

Until `first-candidate.md` and every artifact it cites are written, do not open, print, search
within, import-source-inspect, disassemble, or otherwise read the incumbent implementation below
the public `Mamba2FwdOp` dispatch boundary, including `src/tileops/kernels/mamba/**`, the imported
`CBProducerOp`, `DaCumsumFwdOp`, `SSDChunkStateFwdOp`, `SSDStatePassingFwdOp`, and
`SSDChunkScanFwdOp` bodies, or generated/cached copies of them. Do not use broad searches that
cross these paths. Record all blind-phase source reads and the exact timestamp when blindness
ends.

Before ending the blind phase, the new implementation must:

1. originate from an authored TileFoundry HIR description and a traceable runtime twin;
2. return both FP32 outputs correctly on a small multi-chunk fixture and at least the complete
   `mamba2-2p7b-b1-s2k` BF16 manifest row;
3. record repeated native-CUPTI latency for that manifest row; and
4. save authored HIR, TileFoundry check/analyze results, generated/runtime source paths, errors,
   raw samples, read audit, and limitations in `first-candidate.md`.

Make `first-candidate.md` immutable or record a checksum before reading incumbent bodies. After
that, incumbent inspection is permitted for diagnosis and optimization. The final report must
name every idea adopted from it. Never relabel an incumbent-derived route as TileFoundry-derived.

## Primary public contract

The round targets the primary manifest entry: no `dt_bias`, no `initial_states`, `chunk_size=256`,
and `dt_softplus=True`. Preserve all existing public variants and argument behavior even when the
new optimized specialization applies only to the two primary rows.

| Value | Shape | Dtype / rule |
| --- | --- | --- |
| `x` | `[B,S,H,P]` | FP16 or BF16 |
| `dt` | `[B,S,H]` | FP32 |
| `A` | `[H]` | FP32, generated non-positive |
| `B` | `[B,S,G,N]` | same as `x` |
| `C` | `[B,S,G,N]` | same as `x` |
| `y` | `[B,S,H,P]` | FP32 |
| `final_states` | `[B,H,P,N]` | FP32 |

`S % chunk_size == 0` and `H % G == 0`. Inputs are CUDA tensors. The head-to-group mapping is
`group(h) = floor(h / (H/G))`.

The manifest declares fixed two-output arity, while the current public wrapper retains upstream's
`return_final_states` switch and defaults it to false. The existing benchmark also passes
`dt_bias` despite benchmarking the primary name. For this round's correctness and performance
evidence, invoke the primary contract with `dt_bias=None`, `initial_states=None`, and
`return_final_states=True`; both returned tensors must be produced and timed. Preserve backward
compatibility of the public switch unless a separately justified contract repair is reviewed.

Add focused tests that prevent the current blind spots: primary no-bias semantics, final-state
comparison against the oracle/baseline rather than against itself, exact output dtypes/shapes,
both manifest dtypes, and at least one boundary or tail-sensitive supported case. Existing bias
and initial-state variants must not regress.

## Mathematical definition

Let `R = H/G`, `g(h) = floor(h/R)`, and let all recurrence arithmetic be FP32. For each batch and
head, initialize `state[-1,h,:,:]` to zero. For token `t`:

```text
delta_t[h] = softplus(float(dt[t,h]))
delta_t[h] = clamp(delta_t[h], min=0, max=+inf)
decay_t[h] = exp(float(A[h]) * delta_t[h])

u_t[h,p,n] = delta_t[h] * float(x[t,h,p]) * float(B[t,g(h),n])
state_t[h,p,n] = decay_t[h] * state_(t-1)[h,p,n] + u_t[h,p,n]
y[t,h,p] = sum_n state_t[h,p,n] * float(C[t,g(h),n])
```

Return every `y[t,h,p]` and the final `state_(S-1)`, both as FP32. There is no `D*x`, `z`,
sequence-index reset, variable-length packing, bias, or initial state in the primary entry. A
chunked SSD factorization is equivalent only if it implements exactly this recurrence and
returns the same post-update final state.

Use the existing per-dtype tolerances without loosening them: FP16 `atol=1e-2, rtol=1e-3` and
BF16 `atol=2e-2, rtol=1e-3`. Compare `y` and `final_states` separately and record maximum and mean
absolute error. Use a small direct FP32 recurrence as the independent oracle; the external
baseline is not the only correctness authority.

## TileFoundry authored-HIR description

Author a real TileFoundry description in the shared round directory. It must contain a typed
one-token function and a typed full-sequence ordered recurrence matching this value graph:

```python
@func
def mamba2_step(
    state: Tensor[(B, H, P, N), "f32"],
    x_t: Tensor[(B, H, P), DT],
    dt_t: Tensor[(B, H), "f32"],
    A: Tensor[(H,), "f32"],
    B_t: Tensor[(B, G, N), DT],
    C_t: Tensor[(B, G, N), DT],
) -> tuple[Tensor[(B, H, P), "f32"], Tensor[(B, H, P, N), "f32"]]:
    delta = clamp(softplus(dt_t), min=0.0)
    decay = exp(delta * A)
    B_heads = head_group_gather(B_t, heads=H)
    C_heads = head_group_gather(C_t, heads=H)
    update = delta[..., None, None] * cast(x_t, "f32")[..., None] * cast(B_heads, "f32")[:, :, None, :]
    next_state = decay[..., None, None] * state + update
    y_t = reduce_sum(next_state * cast(C_heads, "f32")[:, :, None, :], axis=-1)
    return y_t, next_state

@func
def mamba2_fwd(x, dt, A, B, C):
    state = zeros((B, H, P, N), "f32")
    y, final_state = ordered_scan(mamba2_step, state, x, dt, B, C, axis=1)
    return y, final_state
```

Adapt syntax and types to the current DSL while preserving FP32 carry, post-update output,
head-group ownership, and sequence order. Exercise `tilefoundry check`, `analyze`, `schedule`,
parser/evaluator, and source-to-source surfaces where applicable. If a current surface cannot
represent the ordered scan or symbolic group gather, validate the largest supported subgraph and
write the smallest concrete failing HIR/TIR reproducer. A search result is not evidence of a gap.

The optimized TileLang runtime may use chunk-local semiseparable matrix products, tensor-core
GEMMs, parallel prefix composition, or a multi-kernel pipeline. Every stage must remain traceable
to the recurrence. It must not materialize a full `[B,S,H,P,N]` FP32 state history. If a
semantics-preserving handwritten TileLang runtime twin is necessary, demonstrate the exact
TileFoundry limitation and measure the workaround cost.

## Manifest distribution

Benchmark every row below; there are exactly two primary dtype/shape rows:

| Label | B | S | H | P | G | N | dtype | chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `mamba2-2p7b-b1-s2k` | 1 | 2,048 | 80 | 64 | 1 | 128 | BF16 | 256 |
| `mamba2-1p3b-b1-s8k` | 1 | 8,192 | 64 | 64 | 1 | 128 | FP16 | 256 |

Tune against both rows, not one. Legacy `Mamba2FwdFixture` smoke/full rows are useful regression
coverage but do not replace or expand the primary SOTA distribution. Do not claim a manifest
result from a fixture that silently supplies `dt_bias`.

## Equivalent official baseline

The installed API is:

```text
mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None,
    dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None,
    dt_softplus=False, dt_limit=(0.0, inf), return_final_states=False,
    return_varlen_states=False)
```

For each primary row call it with the same `x,dt,A,B,C`, `chunk_size=256`,
`dt_softplus=True`, `dt_bias=None`, `initial_states=None`, and
`return_final_states=True`. Leave `D`, `z`, sequence metadata, and variable-length outputs absent.

In this pinned package the native `y` dtype equals `x` (BF16/FP16) while the native final state is
FP32. The TileOPs primary contract requires both outputs FP32, so the timed baseline callable must
return `(native_y.float(), native_final_state.float())`; the `y.float()` conversion stays inside
the timing region. The preflight proves this wrapper runs both rows and agrees with the incumbent
under current tolerances. A pure-PyTorch recurrence remains a correctness oracle only and never
counts as a performance fallback.

The initial ten-sample event medians from `preflight.log` are approximately:

| Row | incumbent contract | official contract |
| --- | ---: | ---: |
| 2K / H80 / BF16 | 0.690 ms | 1.296 ms |
| 8K / H64 / FP16 | 0.982 ms | 1.416 ms |

These are preflight observations, not final SOTA evidence. Re-run in the agent's persistent
container with the final common harness and repeated trials.

## Performance and SOTA contract

Report the final candidate, base-commit incumbent, and contract-equivalent official baseline in
one persistent container, process, GPU, input set, precision contract, warmup policy, and native
CUPTI harness. Preserve both outputs, synchronize correctly, exclude compilation/tuning from
steady-state latency, retain raw samples and `profile_run.log`, and quantify run-to-run noise.

SOTA requires the candidate to be no slower than the official baseline within measured noise on
each of the two rows and to have a strictly lower geometric-mean latency. Also report both
per-row and geometric-mean speedups against the base incumbent. Do not call statistical ties or
regressions SOTA.

Do not move the baseline cast outside timing, omit final-state production, pass bias to only one
path, precompute runtime values, change layouts/dtypes, time fewer stages, dispatch from the
candidate to `mamba_ssm` or the incumbent, or specialize using runtime tensor values. Record the
configuration count, compile-hours/GPU-hours, and selection method. If the target is not reached,
retain a candidate only when independently useful and reviewable.

## Required execution and evidence

1. Complete and seal the two-phase blind evidence before incumbent inspection.
2. Run at least:

   ```bash
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/ops/test_mamba.py -k mamba2
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/test_validate_manifest.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q benchmarks/tests
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q -s benchmarks/ops/bench_mamba2_e2e.py -m smoke
   ```

   Add a focused manifest command or script that measures exactly the two rows above under the
   repaired two-output primary contract, plus broader tests proportional to shared-code changes.
3. Correctness-test both manifest rows and both outputs against the independent recurrence and
   official baseline. Add smaller multi-chunk cases for precise diagnosis.
4. Profile both manifest rows. Attribute launch count, temporary memory, dominant kernels,
   serialization/prefix cost, tensor-core/vector work, and a concrete bottleneck signal. Retain
   traces and exact generated sources.
5. Prove the selected rows execute the new candidate and no forbidden full per-token state
   history or external dispatch is present.
6. Commit, push, open a PR to `tile-ai/TileOPs:main`, and own CI/review until mergeable. Never
   merge it. If no honest PR is warranted, retain the complete blocker report instead.

## TileFoundry gap report

Write `tilefoundry-gaps.md` in the shared round directory. For each demonstrated gap, classify it
as `semantic-blocker`, `lowering/codegen-blocker`, `runtime-blocker`, `performance-blocker`, or
`ergonomics`, and record:

- the smallest current-HIR/TIR/CLI reproducer;
- expected behavior and actual error/output;
- affected rows and measured workaround cost;
- likely owning spec/module without prescribing an unverified fix;
- new, duplicate, or enhancement status.

Compare suspected ordered-scan/indexing issues with round 3's dynamic loop-indexed insertion and
head-group broadcast/repeat issues with round 2's dynamic `RepeatInterleave` gap before declaring
new bugs. Ordinary opportunities to fuse the SSD pipeline are not automatically TileFoundry bugs.

## Final round report

Write `$TRIAL_SOURCE/round-4-mamba2/report.md` containing:

- branch, commits, PR URL/status, container/GPU, and exact versions;
- blind-phase read audit/timestamp, authored HIR validation, checksum, and first candidate;
- exact correctness/test commands and artifact paths;
- raw latency/error/noise tables for both rows, candidate, base incumbent, and equivalent
  official baseline, with geometric means and speedups;
- explicit classification: `measured SOTA`, `improvement without SOTA`, or `no improvement`;
- profiler/temporary-memory/launch evidence, tuning budget, failed approaches, incumbent ideas
  adopted, and remaining risks;
- structured TileFoundry gaps with minimal reproducers and measured workaround costs.

Do not finish before a useful PR is mergeable, or the report proves with reproducible evidence
why no honest PR can be opened.
