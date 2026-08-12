# Round 1 brief: FusedMoEExpertsNopadPersistent3WGFwdOp

## Objective

Produce a new, correct TileLang implementation derived through a TileFoundry authored-HIR
workflow for `FusedMoEExpertsNopadPersistent3WGFwdOp`. Optimize it on the four manifest workloads
for NVIDIA H200. The strongest runnable same-contract external baseline is vLLM 0.19.1 Triton
`fused_experts`. Take a useful result through a mergeable TileOPs PR; if no honest performance PR
is justified, finish with reproducible evidence and concrete TileFoundry/kernel blockers.

You are the only implementation agent for this round. Do not dispatch another agent and do not
edit `$HOST_HOME/TileFoundry`.

## Fixed source and environment

- TileOPs base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`.
- TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
- Persistent worktree container: created by Foreman's post-worktree hook.
- Run all TileOPs Python, tests, benchmarks, and profilers through:

  ```bash
  $HOST_HOME/foreman/local/tileops-container.sh <command> [args...]
  ```

  Calls use `docker exec` in the same named container and the same GPU. `shell` enters it.
- Inside the container: `$CONTAINER_WORKSPACE/tileops` is your writable worktree,
  `$CONTAINER_WORKSPACE/tilefoundry` is read-only, and `$CI_CACHE` persists.
- Use TileFoundry on the host only through
  `$HOST_HOME/TileFoundry/.venv/bin/<tool>`.
- Do not install, upgrade, or replace packages in the pinned image.
- Shared round artifacts belong under
  `$TRIAL_SOURCE/round-1-fused-moe-experts/`.

## Authoritative contract sources

You may read these before the first candidate:

- `src/tileops/manifest/moe.yaml`, entry
  `FusedMoEExpertsNopadPersistent3WGFwdOp` (load structurally through
  `tileops.manifest`, as required by `CLAUDE.md`).
- `workloads/moe.py`, `MoeExpertsWorkload` and its input generation.
- `tests/ops/test_fused_moe_experts.py`, including `_torch_ref_moe` and public contract tests.
- `benchmarks/ops/bench_fused_moe_experts.py` and `benchmarks/benchmark_base.py`.
- `src/tileops/ops/moe/routed_expert/fused_routed_expert.py` for the public Op signature,
  validation, output$CONTAINER_WORKSPACE behavior, and dispatch surface only.
- TileFoundry specs via `tilefoundry spec`, and authored-HIR examples. In particular,
  `examples/granite_4_0_h_small-cuda/model.py:405` is an allowed reference for describing
  runtime-indexed routed experts; it is not the TileOPs incumbent kernel.

Before the first correct TileFoundry-derived candidate and its evidence are written, do not open,
print, search within, import-source-inspect, disassemble, or otherwise read the bodies of the
incumbent TileOPs kernels named by the manifest/Op wrapper. This includes the current MoE
permute/unpermute, grouped-GEMM, persistent 3WG, and fused-activation kernel bodies under
`src/tileops/kernels/`. You may create a new kernel file and use public TileLang documentation or
unrelated kernel families. Record the exact first time this blind phase ends.

After the first candidate is recorded, you may inspect the incumbent for diagnosis and fair
same-process comparison. The final report must identify every idea adopted from it; do not relabel
an incumbent-derived implementation as TileFoundry-derived.

## Public operator contract

Static parameters:

- `T = num_tokens`, `E = num_experts`, `K = top_k`, `H = hidden_size`, `F = ffn_size`.
- `routed_scaling_factor: float = 1.0`.
- `activation` is `silu_and_mul` by default; the public Op also supports exact-erf
  `gelu_and_mul`.

Inputs and output:

| Value | Shape | Dtype / rule |
| --- | --- | --- |
| `output` | `[T, H]` | preallocated, same dtype as `hidden_states` |
| `hidden_states` | `[T, H]` | FP16 or BF16 |
| `w_gate_up` | `[E, 2F, H]` | same as hidden |
| `w_down` | `[E, H, F]` | same as hidden |
| `topk_weights` | `[T, K]` | FP32 |
| `topk_ids` | `[T, K]` | INT32 |
| `expert_map` | `[E]` or `None` | INT32 EP map; manifest path uses `None` |
| `workspace1`, `workspace2` | empty | same FP16/BF16 family; `numel == 0` |

`forward` writes `output` in place and returns `None`. `workspace_shapes()` is `((0,), (0,))`,
`output_shape(T, H)` is `(T, H)`, and weighted reduction is internal. Preserve the entire public
contract. A new production specialization may apply only to BF16/SILU/single-GPU/aligned manifest
shapes, provided non-applicable public cases retain their existing behavior and every measured
candidate path executes the new kernel rather than silently falling back to the incumbent.

## Mathematical definition

For token `t`, route `r`, expert `e = topk_ids[t, r]`, hidden coordinate `h`, and FFN coordinate
`f`:

```text
both[t,r,j] = sum_h float(hidden_states[t,h]) * float(w_gate_up[e,j,h])
gate[t,r,f] = both[t,r,f]
up[t,r,f]   = both[t,r,F+f]
inner[t,r,f] = silu(gate[t,r,f]) * up[t,r,f]
down[t,r,h] = sum_f inner[t,r,f] * float(w_down[e,h,f])
output[t,h] = cast_input_dtype(
    routed_scaling_factor * sum_r float(topk_weights[t,r]) * down[t,r,h]
)
```

For `gelu_and_mul`, replace `silu(gate)` with exact-erf GELU. Repeated expert ids are legal and
each route contributes separately. Single-GPU ids are in `[0, E)`. EP behavior follows the
existing public wrapper: nonlocal expert-map entries must not produce NaN/Inf.

The PyTorch oracle performs FP32 math and accumulation, then casts to the input dtype. Existing
FP16/BF16 correctness uses `atol=1e-2, rtol=1e-2`; fused-vs-unfused activation parity uses
`atol=3e-2, rtol=3e-2`. Do not loosen these tolerances.

## TileFoundry authored-HIR description

Create a real, runnable authored-HIR description under the shared round artifact directory and
validate it with the current TileFoundry evaluator/check surface. The logical default-SILU graph
is:

```python
@func
def routed_experts(
    hidden: Tensor[(T, H), DT],
    topk_weights: Tensor[(T, K), "f32"],
    topk_ids: Tensor[(T, K), "i32"],
    w_gate_up: ConstTensor[(E, 2 * F, H), DT],
    w_down: ConstTensor[(E, H, F), DT],
) -> Tensor[(T, H), DT]:
    flat_ids = tf.reshape(topk_ids, new_shape=(T * K,))
    selected_in = tf.reshape(
        tf.index_select(w_gate_up, flat_ids, dim=0),
        new_shape=(T, K, 2 * F, H),
    )
    hidden_col = tf.reshape(hidden, new_shape=(T, 1, H, 1))
    both = tf.reshape(tf.matmul(selected_in, hidden_col), new_shape=(T, K, 2 * F))
    gate, up = tf.split(both, axis=-1, num_splits=2)
    inner = tf.silu(gate) * up
    selected_down = tf.reshape(
        tf.index_select(w_down, flat_ids, dim=0),
        new_shape=(T, K, H, F),
    )
    down = tf.reshape(
        tf.matmul(selected_down, tf.reshape(inner, new_shape=(T, K, F, 1))),
        new_shape=(T, K, H),
    )
    weighted = tf.cast(down, dtype="f32") * tf.reshape(topk_weights, new_shape=(T, K, 1))
    mixed = tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")
    return tf.cast(mixed * routed_scaling_factor, dtype=DT)
```

Adapt only syntax/type details required by the current spec; preserve this value graph. The HIR is
the semantic reference, not permission to materialize `[T,K,E-sized-weights]` in the fast path.
Use TileFoundry `check`, `analyze`, `schedule`, specs, evaluator, or source-to-source workflow as
applicable to derive and validate the TileLang runtime twin.

If general runtime `IndexSelect`, its lowering, dynamic route-driven grouped execution, FP32
accumulation, fusion, or another required feature is unsupported, do not paper over it. Produce
the smallest current-checkout reproducer and record it under `tilefoundry-gaps.md` using the gap
protocol below. A semantics-preserving handwritten TileLang runtime twin remains allowed only when
the exact TileFoundry limitation and workaround cost are measured.

## Manifest workloads and performance target

All primary rows are BF16, `K=8`, `H=7168`, `F=2048`:

| Label | T | E |
| --- | ---: | ---: |
| `qwen3-235b-decode` | 512 | 128 |
| `qwen3-235b-prefill` | 4096 | 128 |
| `deepseek-v3-decode` | 512 | 256 |
| `deepseek-v3-prefill` | 4096 | 256 |

The manifest roofline accounts `T*K*6*F*H` FLOPs and
`(E*3*F*H + 2*T*H)*elem_bytes` bytes.

The external performance target is vLLM Triton `fused_experts`. The pinned vLLM CUTLASS module
exposes quantized FP8/FP4 paths and has no matching BF16/FP16 entry point; do not compare it as an
equivalent baseline. If this fact changes, prove the matching contract before adding CUTLASS.

For every row, report candidate, base-commit incumbent, and vLLM Triton raw latency from repeated
runs in the same persistent container, process, GPU, input tensors, dtype, warmup policy, and
timing harness. Exclude compile/tuning time. Preserve `profile_run.log` and raw command output.
The SOTA label requires every primary row to be no slower than vLLM within measured noise and a
strictly lower geometric-mean latency. Also report geometric mean and per-row speedup against the
incumbent. A statistically indistinguishable result is not SOTA.

## Required execution and evidence

1. Save `first-candidate.md` before leaving the blind phase. Include authored HIR path and
   validation, new kernel files, correctness command/output, raw latency, and known limitations.
2. Run at least:

   ```bash
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/ops/test_fused_moe_experts.py -m smoke
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/test_validate_manifest.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q benchmarks/tests
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q -s benchmarks/ops/bench_fused_moe_experts.py
   ```

   Add targeted tests for the new kernel and broader suites in proportion to changed shared code.
3. Benchmark all four manifest rows. Ensure the benchmark records the new candidate, the
   incumbent, and vLLM under unambiguous tags; repair the stale CUTLASS import/reporting if needed.
4. Profile the limiting decode and prefill rows. Tie optimizations to measured occupancy, memory
   traffic, launch count, tensor-core utilization, routing imbalance, or another concrete signal.
5. Preserve the public fallback/EP/activation behavior and prove measured rows did not execute an
   incumbent fallback.
6. Commit, push, open a PR to `tile-ai/TileOPs:main`, and own CI/review until mergeable. Never
   merge it. If permissions or a repository policy prevent a PR, record the exact command/error;
   do not claim mergeable.

## TileFoundry gap report

Write `tilefoundry-gaps.md` in the shared round directory. Each gap must be one of
`semantic-blocker`, `lowering/codegen-blocker`, `runtime-blocker`, `performance-blocker`, or
`ergonomics`, with:

- minimal HIR/TIR/current-CLI reproducer;
- expected and actual behavior/error;
- affected workloads and measured workaround cost;
- likely owning spec/module;
- new/duplicate/enhancement classification.

Do not call a merely possible optimization a TileFoundry bug.

## Final round report

Write `$TRIAL_SOURCE/round-1-fused-moe-experts/report.md` containing:

- branch, commits, PR URL/status, container name/GPU, and exact pinned versions;
- blind-phase timestamp and first-candidate summary;
- final correctness commands and raw artifact paths;
- per-workload latency table for candidate/incumbent/vLLM, noise, speedups, and geometric means;
- explicit classification: `measured SOTA`, `improvement without SOTA`, or `no improvement`;
- profiler evidence, tuning budget, failed approaches, incumbent ideas adopted, and residual risk;
- links to authored HIR/check artifacts and structured TileFoundry gaps.

Do not finish before the PR is mergeable or the report proves why no honest PR can be opened.
