# TileFoundry gaps: round 2 GemmW4A16Op

TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
The exact authored graph is `authored_hir.py`; the semantics-preserving static
carrier workaround is `authored_hir_i32_workaround.py`.

## TF-R2-W4-01: unsigned 8-bit tensors are not representable

- Classification: `semantic-blocker`; status: `new`.
- Smallest reproducer: import `authored_hir.py`. Its function has one argument
  annotated `Tensor[(N, K // 2), "u8"]`; the same failure also occurs for the
  UINT8 zero-point argument.
- Expected: import a HIR function whose storage contract includes UINT8 packed
  weights and zero points.
- Actual: `DType.from_name("u8")` raises `ValueError: DType: unknown value
  'u8'; valid: ['bf16', 'bool', 'f16', 'f32', 'f4e2m1', 'f8e8m0',
  'fp8e4m3', 'i32', 'i64']`. Full output is in `hir-import.log`.
- Affected workloads: all six manifest rows. This is part of the public
  operator contract, so substituting signed or floating storage is not a
  deployable lowering.
- Workaround and measured cost: `authored_hir_i32_workaround.py` uses i32
  carriers for both UINT8 inputs and statically specializes `(64,64,128)`. The
  packed-weight and zero-point carriers grow from 4,160 bytes to 16,640 bytes
  for that row (4x storage, +12,480 bytes). The workaround evaluator is
  bitwise equal (`max_abs=0`) and `tilefoundry check` passes, but it cannot be
  used for the TileOPs runtime contract. The delivered runtime is therefore a
  handwritten, graph-traceable TileLang twin.
- Likely owner: `tilefoundry.ir.types.dtype` and the DSL tensor/type validation
  surface. This report does not prescribe a representation or lowering.

## TF-R2-W4-02: RepeatInterleave cannot infer a dynamic repeated extent

- Classification: `semantic-blocker`; status: `new`.
- Smallest reproducer: `authored_hir_i32_dynamic_repro.py`, which removes the
  independent UINT8 blocker by using i32 carriers and calls
  `tf.repeat_interleave(weight_scale, repeats=128, axis=1)` where the input
  extent is `K // 128`. Run `tilefoundry analyze --compute-cost` with
  `--dim M=1 --dim N=8192 --dim K=8192`.
- Expected: infer the repeated axis as `(K // 128) * 128` (and then validate it
  against K), including when the CLI binds K to a concrete extent.
- Actual: `unsupported operand type(s) for *: 'Call' and 'int'`, exit 1. Full
  command output is `hir-i32-dynamic-repro.log`.
- Affected workloads: all six manifest rows when represented by one dynamic
  HIR. The group metadata expansion for both scale and zero uses this
  operation.
- Workaround and measured cost: replace all `DimVar` extents with constants,
  as in `authored_hir_i32_workaround.py`. That makes evaluator/check/analyze
  pass for one shape but requires six separate authored specializations to
  cover the manifest and still does not produce a runtime schedule. The round
  kept one smoke specialization and paid the 24-evaluation manual runtime
  tuning cost described below instead of duplicating six semantic files.
- Likely owner: the type inference relation in
  `tilefoundry.ir.hir.tensor.repeat_interleave` and its symbolic-expression
  shape handling. This report does not assume whether simplification should
  happen in that relation or in a shared symbolic shape layer.

## TF-R2-W4-03: scheduling requires a dense i32 peak for an i32 intermediate

- Classification: `performance-blocker`; status: `new`.
- Smallest reproducer: run `tilefoundry schedule` for
  `authored_hir_i32_workaround.py:GemmW4A16.gemm_w4a16` on
  `nvidia.h200_sxm`, after using the valid static CTA extent 128.
- Expected: schedule the f16-output graph while accounting for the elementwise
  i32 unpack work, or otherwise explain which schedulable operation lacks a
  target fact.
- Actual: `nvidia.h200_sxm states no dense peak rate for i32; it states bf16,
  f16, f32, fp8e4m3`. Full output is in
  `hir-i32-schedule-smoke-3.json`. The same static graph passes evaluator,
  `check`, and `analyze` (`hir-i32-evaluator-smoke.log`,
  `hir-i32-check-smoke-2.json`, and `hir-i32-analyze-smoke.json`).
- Affected workloads: all four M=1 primary rows and both compile-smoke rows;
  nibble unpacking necessarily passes through integer arithmetic.
- Workaround and measured cost: no TileFoundry schedule could be consumed.
  Runtime selection used a controlled handwritten sweep of 6 configurations
  on each of 4 primary rows, 24 shape/config evaluations total, taking
  113.612 seconds (0.0316 H200 GPU-hours). See `tuning.log`.
- Likely owner: `tilefoundry.schedule.partition.facts` together with the H200
  hardware facts in
  `target/cuda/hardware/nvidia_h200_sxm.toml`. It is not clear from this
  reproducer whether the correct ownership is an integer throughput fact or
  scheduling semantics for non-dominant intermediate dtypes.

## Non-gaps retained for context

- A dynamic `Topology("cta", None)` is intentionally rejected by `schedule`
  because scheduling a level requires a static extent. A later CTA extent of
  512 also correctly exceeded the H200 target's stated 132 parallel units.
  Setting the smoke topology to 128 resolved both diagnostics; they are not
  reported as defects.
- `tilefoundry check --inputs random` initially targeted an orchestration
  method with an untyped aggregate argument. Targeting the typed runtime
  function directly resolved it. This was CLI usage, not a capability gap.
