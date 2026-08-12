# Round 2 brief: GemmW4A16Op

## Objective

Produce a new, correct TileLang implementation derived through a TileFoundry authored-HIR
workflow for `GemmW4A16Op`. Optimize the four manifest `M=1` decode workloads on NVIDIA H200.
The strongest runnable same-contract external baseline is vLLM 0.19.1 Marlin W4A16; the
dequantized PyTorch matmul is a correctness oracle only. Take an independently useful result
through a mergeable TileOPs PR. If no honest performance PR is justified, finish with
reproducible evidence and concrete blockers.

This is round 2 of the fixed five-round sequence. Do not use or substitute a GQA operator. You
are the only implementation agent for this round: do not dispatch another agent and do not edit
`$HOST_HOME/TileFoundry`.

## Fixed source and environment

- TileOPs base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`.
- TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
- Pinned image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`, digest
  `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`.
- Persistent worktree container: created once by Foreman's post-worktree hook. Run all TileOPs
  Python, tests, benchmarks, and profiling through:

  ```bash
  $HOST_HOME/foreman/local/tileops-container.sh <command> [args...]
  ```

  Later calls use `docker exec` in that same named container and on the same selected GPU.
  `shell` enters it interactively. Do not create a second ad hoc container.
- Inside the container, `$CONTAINER_WORKSPACE/tileops` is the writable worktree,
  `$CONTAINER_WORKSPACE/tilefoundry` is read-only, and `$CI_CACHE` persists. The common TileOPs Git dir is
  mounted read-only so base-commit source and diffs work; `GIT_OPTIONAL_LOCKS=0` is set.
- Use TileFoundry on the host only through
  `$HOST_HOME/TileFoundry/.venv/bin/<tool>`. Do not rebuild that environment.
- Do not install, upgrade, or replace packages. The container has CUDA 12.9, Python 3.12,
  PyTorch 2.10.0+cu129, TileLang 0.1.11+cu129.gitafcebed1, vLLM 0.19.1, and
  `cupti-python==12.8.0`.
- Native CUPTI timing is the default benchmark path. Do not accept a wall-clock fallback as
  final evidence. NCU performance counters are unavailable on this host with
  `ERR_NVGPUCTRPERM`; retain a single reproducer if needed, then use CUPTI, PyTorch profiler,
  generated-code inspection, and controlled experiments.
- Shared round artifacts belong under
  `$TRIAL_SOURCE/round-2-gemm-w4a16/`.
- Preflight evidence is under `$TRIAL_SOURCE/preflight/`. In particular,
  `round2-marlin-smoke.log` proves both Marlin reduction modes execute and return finite
  `(1, N)` outputs for all four primary shapes. That smoke does not prove numerical equivalence;
  this round must add a common-input correctness check.

## Authoritative contract sources and blind boundary

Before producing the first candidate, read only the relevant contract surfaces:

- Load `GemmW4A16Op` structurally through `tileops.manifest`; its entry is in
  `src/tileops/manifest/gemm.yaml`.
- `workloads/gemm.py`: `quantize_weight_int4`, `GemmW4A16Workload`, input generation, reference,
  and tolerances.
- `src/tileops/ops/gemm.py`: public `GemmW4A16Op` signature, validation, output contract, and
  dispatch surface only.
- `tests/ops/test_gemm.py`: W4A16 fixtures, metadata rejection, and one-sided quantization tests.
- `benchmarks/ops/bench_gemm.py` and `benchmarks/benchmark_base.py`: benchmark contract and the
  current Marlin adapter. The adapter needs an equivalence repair described below.
- TileFoundry specs, tutorial, CLI, authored-HIR examples, and unrelated TileLang kernel families.
- Official vLLM 0.19.1 Marlin packing helpers and custom-op wrapper installed in the fixed image.

Until the first correct TileFoundry-derived candidate and `first-candidate.md` are written, do
not open, print, search within, import-source-inspect, disassemble, or otherwise read
`src/tileops/kernels/gemm_w4a16.py`, a generated copy of it, its compiler cache, or any incumbent
W4A16 kernel body named by the manifest/public wrapper. Do not run broad `rg`/`find`/source dumps
whose results cross into that path. You may create a new kernel file. Record all blind-phase
source-read commands and the exact timestamp at which the blind phase ends.

Before ending the blind phase, save `first-candidate.md` with the authored HIR, current
TileFoundry validation results, generated/runtime-twin source paths, correctness output for at
least one compile-smoke row and `decode-l2-resident-ish`, and repeated raw latency. A candidate
that only produces source, fails to launch, or has not passed numerical comparison is not the
first correct candidate.

After that evidence is immutable, you may inspect the incumbent for diagnosis and a fair
base-commit comparison. The final report must identify every idea adopted from it. Do not
relabel an incumbent-derived optimization as TileFoundry-derived.

## Public operator contract

Static parameter: `group_size=128`; no other group size is supported.

| Value | Shape | Dtype / rule |
| --- | --- | --- |
| `activation` | `[M, K]` | FP16, contiguous |
| `packed_weight` | `[N, K / 2]` | UINT8, contiguous |
| `weight_scale` | `[N, K / 128]` | FP32, contiguous |
| `weight_zero` | `[N, K / 128]` | UINT8, contiguous |
| output | `[M, N]` | FP16 |

`K` is even and divisible by 128. The exact packed/metadata shapes are part of validation; reject
mismatches. Preserve the public Op behavior and kernel caching by `(M, N, K, dtype, group_size)`.
Every benchmarked candidate row must execute the new kernel, not silently dispatch to the
incumbent or an external baseline.

## Quantization and mathematical definition

The workload starts with a logical FP32 weight `W[N,K]`. For each output row `n` and K-group `g`:

```text
group_min = min(min(W[n, 128*g : 128*(g+1)]), 0)
group_max = max(max(W[n, 128*g : 128*(g+1)]), 0)
scale[n,g] = max((group_max - group_min) / 15, 1e-12)             # FP32
zero[n,g]  = clamp(round(-group_min / scale[n,g]), 0, 15)        # UINT8
q[n,k]     = clamp(round(W[n,k] / scale[n,k//128] + zero[n,k//128]), 0, 15)
```

Packing is little-nibble-first along K:

```text
packed_weight[n,j] = q[n,2*j] | (q[n,2*j+1] << 4)
dequant[n,k] = float(q[n,k] - zero[n,k//128]) * scale[n,k//128]
W_ref = cast_fp16(dequant)
output = activation @ transpose(W_ref)
```

The PyTorch oracle performs the final matmul with FP16 operands using PyTorch's CUDA semantics.
Existing correctness uses `atol=7e-2`, `rtol=5e-2`; do not loosen it. Preserve low/high nibble
ordering, unsigned zero points, group boundaries, one-sided groups (zero point 0 or 15), and the
FP16 cast before the reference matmul.

## TileFoundry authored-HIR description

Create a real, runnable authored-HIR description under the shared round directory and exercise
the current TileFoundry `check`, `analyze`, `schedule`, specs, evaluator, or source-to-source
workflow as applicable. The semantic graph is:

```python
@func
def gemm_w4a16(
    activation: Tensor[(M, K), "f16"],
    packed_weight: Tensor[(N, K // 2), "u8"],
    weight_scale: Tensor[(N, K // 128), "f32"],
    weight_zero: Tensor[(N, K // 128), "u8"],
) -> Tensor[(M, N), "f16"]:
    packed_i32 = tf.cast(packed_weight, dtype="i32")
    low = tf.mod(packed_i32, 16)
    high = tf.floor_div(packed_i32, 16)
    q_i32 = tf.reshape(tf.stack((low, high), axis=-1), new_shape=(N, K))

    scale_expanded = tf.reshape(
        tf.broadcast_to(
            tf.reshape(weight_scale, new_shape=(N, K // 128, 1)),
            new_shape=(N, K // 128, 128),
        ),
        new_shape=(N, K),
    )
    zero_expanded = tf.reshape(
        tf.broadcast_to(
            tf.reshape(tf.cast(weight_zero, dtype="f32"),
                       new_shape=(N, K // 128, 1)),
            new_shape=(N, K // 128, 128),
        ),
        new_shape=(N, K),
    )
    dequant_f32 = (tf.cast(q_i32, dtype="f32") - zero_expanded) * scale_expanded
    dequant_f16 = tf.cast(dequant_f32, dtype="f16")
    return tf.matmul(activation, tf.transpose(dequant_f16, axes=(1, 0)))
```

Adapt syntax or type names only as required by the current spec, preserving this value graph and
the even/odd K interleave. The logical HIR may materialize dequantization for semantic checking;
the optimized TileLang runtime twin must fuse unpack, dequantization, and matmul and must not
materialize an `[N,K]` dequantized matrix during timed execution.

If stack/interleave, UINT8-to-integer arithmetic, group metadata broadcast, FP32 dequantization,
the FP16 rounding point, fusion, lowering, or scheduling cannot be represented or lowered, do
not hide it. Record the smallest current-checkout reproducer and measured workaround cost using
the gap protocol. A semantics-preserving handwritten TileLang runtime twin is allowed only when
it remains traceable to the authored HIR and the exact TileFoundry limitation is demonstrated.

## Manifest workloads

Correctness covers every manifest row:

| Label | M | N | K | Role |
| --- | ---: | ---: | ---: | --- |
| `compile-smoke-square-64x64x128` | 64 | 64 | 128 | compile/smoke |
| `compile-smoke-rect-128x256x256` | 128 | 256 | 256 | compile/smoke |
| `decode-l2-resident-ish` | 1 | 8192 | 8192 | primary performance |
| `decode-hbm-streaming-threshold` | 1 | 8192 | 16384 | primary performance |
| `decode-non-power2-low-cta` | 1 | 7168 | 20480 | primary performance |
| `decode-long-k-pressure` | 1 | 8192 | 81920 | primary performance |

Also retain the existing `(M,N,K)=(1,512,512)` and `(16,1024,1024)` fixture coverage, metadata
shape rejection, and one-sided quantization behavior. Add targeted nibble/group-boundary tests if
the existing tests do not independently detect a swap or wrong metadata expansion.

The performance distribution consists of the four `M=1` primary rows. Do not average the two
compile-smoke rows into the SOTA result or specialize only one favored decode row.

## Equivalent Marlin baseline

For each primary row, derive candidate/incumbent inputs and Marlin inputs from the same
`activation`, logical 4-bit `q[N,K]`, FP32 `scale[N,K/128]`, and UINT8 `zero[N,K/128]`:

1. Keep the TileOPs little-nibble-first `packed_weight[N,K/2]` contract unchanged.
2. Use vLLM's installed, official Marlin repack/permutation/zero-point helpers to construct
   `b_q_weight`, `b_scales`, and `b_zeros` in the layouts required by
   `vllm._custom_ops.marlin_gemm`. Do not invent an unverified ad hoc permutation.
3. Perform all conversion/repacking and workspace allocation outside timed regions.
4. Numerically compare both `use_fp32_reduce=False` and `True` outputs with the same dequantized
   PyTorch oracle and the operator tolerance. A mode that fails equivalence is not runnable for
   this contract. Record max absolute/relative error.
5. The fastest correctness-passing reduction mode for each row is the external target. Report
   both modes separately; do not choose by one row globally unless the data supports that choice.

The current benchmark's independent random `qweight/scales/zeros` and shape/finite-only check is
insufficient and must not be used as SOTA evidence. Repair the benchmark adapter and add focused
tests for the conversion. The dequantized PyTorch matmul remains a correctness oracle and may be
reported for context, but it is not a competing performance baseline.

## Performance and SOTA contract

For every primary row, report candidate, base-commit incumbent, and every correctness-passing
Marlin mode from repeated runs in the same persistent container, process, GPU, exact logical
inputs, precision contract, warmup policy, and CUPTI timing harness. Synchronize correctly and
exclude compilation, tuning, repacking, dequantized-oracle construction, and workspace setup from
steady-state latency. Preserve raw output and `profile_run.log`.

Quantify run-to-run noise. The SOTA label requires the candidate to be no slower than the fastest
runnable equivalent Marlin result within measured noise on every primary row, and to have a
strictly lower geometric-mean latency. Also report per-row and geometric-mean speedups against the
base incumbent. Statistically indistinguishable results or a regression on one primary row are
not SOTA.

Tune against all four primary rows. Do not remove slow rows, alter layouts, loosen tolerances,
precompute runtime work, use a different logical weight, time less work, or call Marlin/the
incumbent from the delivered kernel. Record the number of configurations, compile-hours/GPU-hours,
and selection method so the tuning budget is auditable.

## Required execution and evidence

1. During the blind phase, author and validate the HIR, build a real TileLang runtime twin, and
   save `first-candidate.md` as specified above before inspecting the incumbent.
2. Run at least:

   ```bash
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/ops/test_gemm.py -k 'W4A16 or w4a16'
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/test_validate_manifest.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q benchmarks/tests
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q -s benchmarks/ops/bench_gemm.py -k w4a16
   ```

   Add focused Marlin conversion tests and broader suites in proportion to shared-code changes.
3. Benchmark all four primary rows with candidate, base incumbent, equivalent Marlin modes, and
   oracle checks under unambiguous tags. Correctness-test all six manifest rows.
4. Profile at least `decode-l2-resident-ish` and `decode-long-k-pressure`. Tie optimization
   decisions to measured launch count, memory traffic proxies, tensor-core/vector utilization,
   occupancy/resource estimates, generated instructions, or controlled ablations.
5. Prove the measured candidate rows did not execute an incumbent fallback and that the runtime
   did not materialize the full dequantized weight.
6. Commit, push, open a PR to `tile-ai/TileOPs:main`, and own CI/review until mergeable. Never
   merge it. If permissions or policy prevent a PR, retain the exact command/error.

If SOTA is not reached after substantive profiling, preserve the best correct candidate only if
it is independently useful and reviewable. Do not open a performance PR for dead code, a slower
replacement without a justified tradeoff, or benchmark-only manipulation. Either way, finish the
round report with measured blocker evidence.

## TileFoundry gap report

Write `tilefoundry-gaps.md` in the shared round directory. Classify each demonstrated gap as
`semantic-blocker`, `lowering/codegen-blocker`, `runtime-blocker`, `performance-blocker`, or
`ergonomics`, and include:

- the smallest HIR/TIR/current-CLI reproducer or generated description;
- expected behavior and actual error/output;
- affected workloads and measured workaround cost;
- likely owning spec/module, without prescribing an unverified fix;
- new, duplicate, or enhancement classification.

Do not call an ordinary optimization opportunity a TileFoundry bug.

## Final round report

Write `$TRIAL_SOURCE/round-2-gemm-w4a16/report.md` containing:

- branch, commits, PR URL/status, container name/GPU, and exact pinned versions;
- blind-phase read audit, timestamp, authored HIR path/validation, and first-candidate result;
- final correctness commands and raw artifact paths;
- per-workload raw latency for candidate, base incumbent, both Marlin modes, and PyTorch oracle
  context; max errors, noise, speedups, and geometric means;
- explicit classification: `measured SOTA`, `improvement without SOTA`, or `no improvement`;
- profiler evidence, tuning budget, failed approaches, incumbent ideas adopted, and residual risk;
- structured TileFoundry gaps with minimal reproducers and measured workaround costs.

Do not finish before a useful PR is mergeable, or the report proves why no honest PR can be
opened.
