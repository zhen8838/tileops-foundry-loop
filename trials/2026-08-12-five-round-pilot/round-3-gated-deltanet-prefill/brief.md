# Round 3 brief: GatedDeltaNetPrefillFwdOp

## Objective

Produce a correct TileLang implementation derived through a TileFoundry authored-HIR workflow
for `GatedDeltaNetPrefillFwdOp`, then optimize it across the complete production BTHD manifest on
NVIDIA H200. The strongest runnable external baseline is FLA 0.4.2
`chunk_gated_delta_rule`. Take an independently useful result through a mergeable TileOPs PR; if
no honest performance PR is justified, finish with reproducible evidence and concrete blockers.

This is round 3 of the fixed five-round sequence. Do not use or substitute a GQA operator. You
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

  Later calls use `docker exec` in the same named container and on the same selected GPU.
  `shell` enters it interactively. Do not create another ad hoc container.
- Inside the container, `$CONTAINER_WORKSPACE/tileops` is writable, `$CONTAINER_WORKSPACE/tilefoundry` is read-only,
  and `$CI_CACHE` persists. The common TileOPs Git dir is mounted read-only so base-commit source
  and diffs work; `GIT_OPTIONAL_LOCKS=0` is set.
- Use TileFoundry on the host only through
  `$HOST_HOME/TileFoundry/.venv/bin/<tool>`. Do not rebuild that environment.
- Do not install, upgrade, or replace packages. The image has CUDA 12.9, Python 3.12, PyTorch
  2.10.0+cu129, TileLang 0.1.11+cu129.gitafcebed1, FLA 0.4.2, and
  `cupti-python==12.8.0`.
- Native CUPTI timing is required for final latency. NCU performance counters are unavailable on
  this host with `ERR_NVGPUCTRPERM`; preserve one existing reproducer and use CUPTI, PyTorch
  profiler, generated-source inspection, and controlled ablations.
- Shared artifacts belong under
  `$TRIAL_SOURCE/round-3-gated-deltanet-prefill/`.
- Shared preflight evidence is under `$TRIAL_SOURCE/preflight/`.
  `round3-fla-production-smoke.log` proves FLA agrees with the small recurrence oracle for FP16
  and BF16 and executes the largest BF16 manifest shape with finite outputs.

## Authoritative sources and blind boundary

Before producing the first correct candidate, read only these contract surfaces:

- Load `GatedDeltaNetPrefillFwdOp` structurally through `tileops.manifest`; its entry is in
  `src/tileops/manifest/linear_attention.yaml`.
- `workloads/linear_attention.py`: `GatedDeltaNetPrefillFwdWorkload`, input generation, reference,
  and gated recurrence helpers.
- `src/tileops/ops/gated_deltanet.py`: public prefill Op signature, validation, layout handling,
  output shapes, cache key, and dispatch surface only.
- `tests/ops/test_gated_deltanet_prefill.py`: correctness, dtype, chunk, layout, and transition
  tests.
- `benchmarks/ops/bench_gated_deltanet_prefill.py` and `benchmarks/benchmark_base.py`.
- The installed FLA 0.4.2 wrapper and source for `chunk_gated_delta_rule`.
- TileFoundry specs/tutorial/CLI and authored HIR examples. The shipped
  `tests/models/qwen3_5_35b_a3b/model.py` `delta_step` is an allowed semantic reference for one
  recurrent step; it is not the TileOPs incumbent.
- Unrelated TileLang kernel families needed to learn repository conventions.

Until `first-candidate.md` and all evidence it cites are written, do not open, print, search
within, import-source-inspect, disassemble, or otherwise read
`src/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py`, a generated/cached copy of it, or
any incumbent prefill kernel body named by the manifest/public wrapper. Do not run broad source
searches whose results cross into that path. Record every blind-phase source-read command and the
exact timestamp at which the blind phase ends.

Before ending the blind phase, the new implementation must pass both outputs against the oracle
on at least one small fixture and the `S=4096,H=16,D=128` manifest row, and it must have repeated
native-CUPTI latency for that manifest row. Save the authored HIR, current TileFoundry validation,
new runtime-twin source paths, errors, raw samples, and known limitations in
`first-candidate.md`. Source generation, compilation alone, or checking only `o` is insufficient.

After that evidence is immutable, you may inspect the incumbent for diagnosis and fair
base-commit comparison. Identify every adopted incumbent idea in the final report. Do not
relabel an incumbent-derived path as TileFoundry-derived.

## Public operator contract

Parameters:

- `layout` is `"bthd"` by default and may be `"bthd"` or `"bhtd"`.
- `chunk_size` may be explicit or `None`. When omitted, the public Op chooses 128 if
  `batch * heads <= 8` and `S` is divisible by 128, otherwise 64.
- `tune` and `kernel_map` retain their public behavior.

For production `layout="bthd"`:

| Value | Shape | Dtype / rule |
| --- | --- | --- |
| `q` | `[B, S, H, DK]` | FP16, BF16, or FP32 |
| `k` | `[B, S, H, DK]` | same as q |
| `v` | `[B, S, H, DV]` | same as q |
| `g` | `[B, S, H]` | same as q; log forget increments |
| `beta` | `[B, S, H]` | same as q |
| `o` | `[B, S, H, DV]` | same as q |
| `final_state` | `[B, H, DK, DV]` | same as q |

For `layout="bhtd"`, sequence and head axes are swapped in q/k/v/g/beta/o; final state remains
`[B,H,DK,DV]`. Inputs must be CUDA tensors with exact matching shapes/dtypes. The public Op makes
inputs contiguous before dispatch. `S` must be divisible by the selected chunk size.

A production specialization may target only the manifest contract (`B=1`, BTHD, `DK=DV=128`,
chunk 64, FP16/BF16), provided every non-applicable public case retains its existing behavior.
Every measured manifest row must prove it executes the new TileFoundry-derived runtime, never an
incumbent fallback or FLA.

## Mathematical definition

The inputs contain raw per-token log forget increments. For each batch `b`, head `h`, and token
`t`, maintain an FP32 state `H_t[DK,DV]` from zero. Let all arithmetic below be FP32:

```text
H_-1 = zeros([DK,DV])
alpha_t = exp(float(g[b,t,h]))
q_t = float(q[b,t,h,:])
k_t = float(k[b,t,h,:])
v_t = float(v[b,t,h,:])
beta_t = float(beta[b,t,h])

old_t = transpose(k_t) @ H_(t-1)                       # [DV]
delta_t = beta_t * (v_t - alpha_t * old_t)             # [DV]
H_t = alpha_t * H_(t-1) + outer(k_t, delta_t)           # [DK,DV]
o[b,t,h,:] = q_t @ H_t                                  # [DV]
```

Return `cast_input_dtype(o)` and `cast_input_dtype(H_(S-1))`. This is the zero-initial-state
sequential definition. The chunked WY/block-solve formulation is an algebraically equivalent
implementation, not a different contract. TileOPs uses score scale 1.0; do not apply
`1/sqrt(DK)` or q/k normalization. For each chunk, the workload reference's chunk-local
`cumsum(g)` represents products of the same per-token `alpha_t` values.

The existing tolerances are FP32 `atol=rtol=1e-2`, FP16 `5e-2`, and BF16 `1e-1`; do not loosen
them. Validate both `o` and `final_state`, including layout parity and chunk sizes 32/64. Fix or
augment any test that accidentally compares a final state to itself rather than the other path.

## TileFoundry authored-HIR description

Author a real TileFoundry description under the shared round directory. At minimum it must
contain a typed one-step function matching this value graph and a typed prefill/scan contract:

```python
@func
def delta_step(
    state: Tensor[(B, H, DK, DV), "f32"],
    q_t: Tensor[(B, H, DK), DT],
    k_t: Tensor[(B, H, DK), DT],
    v_t: Tensor[(B, H, DV), DT],
    g_t: Tensor[(B, H), DT],
    beta_t: Tensor[(B, H), DT],
) -> tuple[Tensor[(B, H, DV), "f32"], Tensor[(B, H, DK, DV), "f32"]]:
    alpha = tf.exp(tf.cast(g_t, dtype="f32"))
    qf, kf, vf = tf.cast(q_t, "f32"), tf.cast(k_t, "f32"), tf.cast(v_t, "f32")
    old = tf.matmul(tf.reshape(kf, (B,H,1,DK)), state).reshape(B,H,DV)
    delta = tf.cast(beta_t, "f32").reshape(B,H,1) * (
        vf - alpha.reshape(B,H,1) * old
    )
    next_state = alpha.reshape(B,H,1,1) * state + outer(kf, delta)
    out = tf.matmul(tf.reshape(qf, (B,H,1,DK)), next_state).reshape(B,H,DV)
    return out, next_state

@func
def gated_deltanet_prefill(q, k, v, g, beta):
    state = zeros((B,H,DK,DV), "f32")
    # Ordered scan over BTHD sequence axis using delta_step as the carry update.
    out_f32, final_state_f32 = ordered_scan(delta_step, state, q, k, v, g, beta, axis=1)
    return cast(out_f32, DT), cast(final_state_f32, DT)
```

Adapt exact syntax/types to the current DSL while preserving ordered state dependence, FP32
carry, post-update output, scale 1.0, and BTHD layout. Exercise `tilefoundry check`, `analyze`,
`schedule`, evaluator, parser, or source-to-source surfaces as applicable. If current HIR has no
ordered scan/carry representation, author and validate the one-step HIR, then create the smallest
multi-step or loop reproducer demonstrating the missing semantic/lowering surface. Do not claim
the feature is absent from a text search alone.

The optimized TileLang runtime may use chunked WY transforms, block solves, parallel prefix
composition, or a multi-kernel pipeline, but every stage must remain traceable to this recurrence.
It must not materialize an avoidable dense object that makes long-context rows impractical. If a
semantics-preserving handwritten TileLang runtime twin is necessary, demonstrate the exact
TileFoundry limitation and measure the workaround cost.

## Manifest distribution

All rows have `B=1`, BTHD layout, `DK=DV=128`, chunk size 64, and both FP16/BF16:

| Label family | S | H |
| --- | ---: | ---: |
| `fallback-gdn-prefill-b1-s4k-h16-d128-bthd` | 4,096 | 16 |
| `qwen35-...-s32k-h16` | 32,768 | 16 |
| `qwen35-...-s64k-h16` | 65,536 | 16 |
| `qwen35-...-s128k-h16` | 131,072 | 16 |
| `qwen35-...-s32k-h32` | 32,768 | 32 |
| `qwen35-...-s64k-h32` | 65,536 | 32 |
| `qwen35-...-s128k-h32` | 131,072 | 32 |
| `qwen35-...-s32k-h48` | 32,768 | 48 |
| `qwen35-...-s64k-h48` | 65,536 | 48 |
| `qwen35-...-s128k-h48` | 131,072 | 48 |
| `qwen35-...-s32k-h64` | 32,768 | 64 |
| `qwen35-...-s64k-h64` | 65,536 | 64 |
| `qwen35-...-s128k-h64` | 131,072 | 64 |

Treat all 26 dtype/shape rows as the performance distribution: benchmark and report every row,
and do not exclude the 4k fallback row from the SOTA criterion. Tune across the whole
distribution, not just one sequence length or head count.

## Equivalent FLA baseline

The installed API is:

```text
chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None,
                       output_final_state=False, ...)
```

For every manifest row, use the exact same already-BTHD inputs and call with `scale=1.0`,
`initial_state=None`, `output_final_state=True`, and no internal q/k L2 normalization. FLA defaults
to `1/sqrt(DK)` when scale is omitted, so an omitted scale is not equivalent.

FLA returns `o` in the input dtype but returns `final_state` as FP32 in the pinned version. The
TileOPs contract requires the state in the input dtype. The timed external callable must therefore
include `final_state.to(q.dtype)` and return the same two-output contract. Do not move that cast
outside timing. Compare both returned tensors with the same oracle before the FLA result is
eligible as a baseline, recording maximum errors separately.

The preflight already proves the raw FLA implementation executes the largest BF16 shape. Any
future unrunnable declaration requires an exact import/runtime reproducer and environment facts;
do not fall back to the pure-Torch recurrence as a performance target.

## Performance and SOTA contract

For all 26 rows, report final candidate, base-commit incumbent, and equivalent FLA from repeated
runs in the same persistent container, process, GPU, exact tensors, dtype, warmup policy, and
CUPTI harness. Preserve both outputs, synchronize correctly, and exclude compilation/tuning from
steady-state latency. Input layout conversion is unnecessary because the manifest and FLA are
both BTHD. Preserve raw output and `profile_run.log`.

Quantify run-to-run noise. SOTA requires the candidate to be no slower than equivalent FLA within
measured noise on every one of the 26 rows and to have a strictly lower geometric-mean latency.
Also report per-row and geometric-mean speedups against the base incumbent. A statistically
indistinguishable result or any primary-row regression is not SOTA.

Do not remove slow rows, loosen tolerances, use different inputs, time only one output, omit the
state cast from FLA, precompute runtime work, or call FLA/the incumbent from the candidate. Record
the configuration count, compile-hours/GPU-hours, and selection method. If SOTA is not reached,
retain a candidate only if it is independently useful and reviewable.

## Required execution and evidence

1. Complete the two-phase blind workflow and save `first-candidate.md` before reading the
   incumbent.
2. Run at least:

   ```bash
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/ops/test_gated_deltanet_prefill.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/test_validate_manifest.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q benchmarks/tests
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q -s benchmarks/ops/bench_gated_deltanet_prefill.py
   ```

   Add focused FLA contract tests and broader suites in proportion to any shared-code changes.
3. Correctness-test every manifest row for both outputs/dtypes. If full PyTorch recurrence is too
   slow for long contexts, use cross-checked chunk decomposition, FLA equivalence, state-prefix
   invariants, and smaller exact oracle cases; do not silently omit correctness.
4. Profile at least the `S=4096,H=16` and `S=131072,H=64` BF16 rows. Attribute launch count,
   temporary memory, matrix-core/vector work, occupancy/resource estimates, state-scan
   serialization, or another concrete limiting signal. Save profiler traces and generated code.
5. Prove all measured rows execute the new candidate and that no forbidden dense per-sequence
   state history or other asymptotically excessive temporary is created.
6. Commit, push, open a PR to `tile-ai/TileOPs:main`, and own CI/review until mergeable. Never
   merge it. If no honest PR is warranted, retain the complete blocker report instead.

## TileFoundry gap report

Write `tilefoundry-gaps.md` in the shared round directory. Classify each demonstrated gap as
`semantic-blocker`, `lowering/codegen-blocker`, `runtime-blocker`, `performance-blocker`, or
`ergonomics`, with:

- the smallest current-HIR/TIR/CLI reproducer;
- expected behavior and actual error/output;
- affected rows and measured workaround cost;
- likely owning spec/module, without prescribing an unverified fix;
- new, duplicate, or enhancement status.

Check against the round 1 runtime `IndexSelect` and round 2 UINT8/dynamic-repeat/i32-schedule gaps
before calling anything a duplicate. Do not call ordinary scan optimization work a TileFoundry
bug.

## Final round report

Write `$TRIAL_SOURCE/round-3-gated-deltanet-prefill/report.md` with:

- branch, commits, PR URL/status, container/GPU, and exact versions;
- blind-phase read audit, timestamp, authored HIR validation, and first-candidate result;
- all correctness commands and raw artifact paths;
- 26-row latency/error/noise tables for candidate, base incumbent, and contract-equivalent FLA,
  plus geometric means and speedups;
- explicit classification: `measured SOTA`, `improvement without SOTA`, or `no improvement`;
- profiler evidence, temporary-memory/launch analysis, tuning budget, failed approaches,
  incumbent ideas adopted, and remaining risk;
- structured TileFoundry gaps with minimal reproducers and measured workaround costs.

Do not finish before a useful PR is mergeable, or the report proves why no honest PR can be
opened.
