# Round 1 report: fused MoE experts

## CUDA 13.2 revalidation (2026-08-12)

PR #1886 was revalidated after the shared persistent-container environment
moved to the official CUDA 13.2 development image. The PR source needed no
compatibility or correctness fix. Its single commit was rebased unchanged onto
the runner migration at `0d75579`; the resulting PR commit is `5fbb6a8`.

### Exact environment

- Image tag: `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`
- Image digest and container image ID:
  `sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`
- Container: `tileops-d4fbbea95c75`, schema 3, physical GPU index 1
- GPU: NVIDIA H200, SM90, UUID
  `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`
- Driver: 595.71.05
- CUDA toolkit: 13.2.1 (`nvcc` 13.2.78); PyTorch CUDA runtime: 13.2
- Python: 3.12.13
- PyTorch: `2.13.0+cu132`
- TileLang: `0.1.11+cu132.gitafcebed1`
- Triton: 3.7.1
- vLLM: 0.27.1
- cupti-python: 13.2.0; cuda-bindings: 13.3.1
- apache-tvm-ffi: 0.1.11
- FlashInfer: 0.6.16.post3; FLA: 0.5.2
- mamba-ssm: 2.3.2.post1; DeepGEMM: `2.1.1+c9f8b34`

Both `scripts/ci/verify_runner_image.py` and
`scripts/ci/verify_runtime_stack.py` passed after rebasing onto the migration
commit. Before that rebase, both correctly rejected the checkout because the
old base still expected CUDA 12.9; that was stale base metadata, not a PR
failure.

### Correctness

One serialized H200 run covered the full supported PR surface, including the
nightly GELU fused-parity case that the original smoke-filtered run excluded:

```text
python -m pytest -q \
  tests/ops/test_fused_moe_experts.py \
  tests/kernels/test_moe_grouped_gemm_3wg_fused_act.py \
  tests/test_validate_manifest.py benchmarks/tests \
  tests/test_workload_placement.py
189 passed, 1 warning in 274.99s
```

Raw output and JUnit are `evidence-cu132/correctness.log` and
`evidence-cu132/correctness.xml`. The warning is the intentional manifest
name-mismatch fixture in `test_single_class_file_rejects_mismatched_name`.
The four changed files also pass Ruff and `git diff --check`.

### Native-CUPTI benchmark

The benchmark ran all four manifest workloads in one pytest process. For each
workload it used the same tensors for `tileops-fused-act`, `base-incumbent`,
and vLLM Triton, then measured the implementations in forward and reverse
order. The repository harness rotated input addresses, flushed L2, excluded
compilation, and attributed the discovered kernel sequence with native CUPTI.
`TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0` made any fallback fatal. All 12 rows
report `timing=cupti`.

Raw report, stdout and JUnit are `evidence-cu132/profile_run.log`,
`evidence-cu132/benchmark.log`, and `evidence-cu132/benchmark.xml`.

| workload | candidate | p10-p90 | incumbent | p10-p90 | vLLM | p10-p90 | inc speedup | vLLM/candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-235b-decode | 3.0439 | 3.0214-6.3864 | 3.1617 | 3.1569-3.1727 | 2.9422 | 2.9214-5.9032 | 1.0387x | 0.9666x |
| qwen3-235b-prefill | 6.0151 | 5.6636-9.1768 | 5.7425 | 5.7262-5.7956 | 6.7984 | 6.7619-9.6832 | 0.9547x | 1.1302x |
| deepseek-v3-decode | 5.8042 | 5.7704-9.0679 | 6.1058 | 6.0970-6.1191 | 5.5988 | 5.5811-9.0702 | 1.0520x | 0.9646x |
| deepseek-v3-prefill | 8.2100 | 7.9039-11.4620 | 8.1078 | 8.0316-8.2031 | 8.5403 | 8.4882-12.1867 | 0.9876x | 1.0402x |
| geometric mean | 5.4349 | - | 5.4754 | - | 5.5611 | - | 1.0075x | 1.0232x |

The CUPTI p90 values contain occasional long inter-kernel gaps. Medians are
stable and remain the decision statistic; incumbent rows, in particular, have
tight p10-p90 ranges. The classification remains **improvement without
SOTA**: geometric-mean candidate latency is 0.75% below incumbent and 2.32%
below vLLM, but candidate loses to incumbent on both prefill rows and to vLLM
on both decode rows. The public default remains unchanged.

### CUDA 12.9 versus CUDA 13.2

The table compares the original CUDA 12.9 medians with the new CUDA 13.2
medians. Negative percentages are faster. This is a cross-stack and
cross-timer comparison: the old image lacked cupti-python and used CUDA events,
whereas the new official image uses native CUPTI, so small deltas are not
attributed to the CUDA migration.

| workload | candidate old/new (delta) | incumbent old/new (delta) | vLLM old/new (delta) |
| --- | ---: | ---: | ---: |
| qwen3-235b-decode | 3.0369 / 3.0439 (+0.23%) | 3.1552 / 3.1617 (+0.21%) | 3.0022 / 2.9422 (-2.00%) |
| qwen3-235b-prefill | 6.0420 / 6.0151 (-0.45%) | 5.7479 / 5.7425 (-0.09%) | 7.3834 / 6.7984 (-7.92%) |
| deepseek-v3-decode | 5.7930 / 5.8042 (+0.19%) | 6.0943 / 6.1058 (+0.19%) | 5.6403 / 5.5988 (-0.74%) |
| deepseek-v3-prefill | 8.1298 / 8.2100 (+0.99%) | 8.1776 / 8.1078 (-0.85%) | 8.6845 / 8.5403 (-1.66%) |
| geometric mean | 5.4219 / 5.4349 (+0.24%) | 5.4830 / 5.4754 (-0.14%) | 5.7403 / 5.5611 (-3.12%) |

TileOps candidate and incumbent move by at most 0.99% per row and 0.24% in
geometric mean, with no systematic regression. The only material delta is
vLLM Qwen prefill. A same-contract native-CUPTI confirmation measured
candidate/incumbent/vLLM at 6.0258/5.7274/6.9887 ms, reproducing the vLLM
improvement relative to 7.3834 ms while keeping the TileOps rows stable. Its
raw artifacts are `evidence-cu132/qwen-prefill-confirm-profile_run.log`,
`evidence-cu132/qwen-prefill-confirm.log`, and
`evidence-cu132/qwen-prefill-confirm.xml`.

The investigation compared vLLM 0.19.1 and 0.27.1 source. Both choose the same
BF16 Triton defaults for these shapes (`BM64` at T=512, `BM128` at T=4096,
`BN128/BK64`, 8 warps, stage 3, group 1). vLLM 0.27.1's tensor-descriptor path
defaults off on CUDA and its swap-AB path does not apply to these M tiles.
Therefore the observed improvement is real for the new official environment,
but cannot be assigned to one mechanism: CUDA, PyTorch, Triton, vLLM, and the
timing method all changed together. It does not expose a PR compatibility or
correctness issue and does not justify a PR source change.

### CI and review status

The rebased commit `5fbb6a81af8a9426b9139cc024549696b940c0ff` is pushed and
matches `origin/perf/tileops-r1-fused-moe`. All applicable PR checks passed,
including pre-commit, benchmark and compile contracts, packaging, gitleaks,
actionlint, security policy, and GPU smoke. The CUDA 13.2 GPU job completed in
5m35s: the repository smoke shard reported 2,915 passed and 14 skipped, and
the diff-scoped changed-test run reported 33 passed, both with zero failures.
The downloaded job artifact is preserved under `evidence-cu132/ci/`.

GitHub reports the source as `mergeable=MERGEABLE` with no conflict, pending
check, review comment, or requested change. `mergeStateStatus=BLOCKED` is
solely the protected `main` rule requiring one approving human review and
approval of the last push (`required_approving_review_count=1`,
`require_last_push_approval=true`). There are currently zero reviews and zero
comments. No approval bypass or merge was attempted.

## Original CUDA 12.9 result

Classification: **improvement without SOTA**.

The production PR keeps the public operator contract unchanged and improves
the existing opt-in fused-activation kernel: it avoids a redundant output
memset and selects the measured `BM64/stage2` template when routed rows per
expert are at most 16. The benchmark now compares this path with the base
default pipeline and vLLM Triton in one drift-balanced run, owns its FP32
workload oracle, and proves the measured candidate did not silently fall back.

The new authored-HIR-derived direct TileLang implementation was correct but
unusable at 953.0823 ms on the first decode row. It is preserved as evidence,
not shipped as dead production code.

## Original CUDA 12.9 source and environment

- Branch: `perf/tileops-r1-fused-moe`
- TileOPs base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`
- TileOPs commit: `a7dfcc752bca1236267b5832e603e93bed68b642`
- PR: https://github.com/tile-ai/TileOPs/pull/1886
- PR status: open, `mergeable=MERGEABLE`, all CI checks green. GitHub's
  `mergeStateStatus=BLOCKED` is solely the protected branch's one-human-review
  rule (`required_approving_review_count=1`, last-push approval required), not
  a conflict or failing check. No review bypass or merge was attempted.
- Container: `tileops-d4fbbea95c75`, persistent GPU index 1
- GPU: NVIDIA H200, SM90, UUID
  `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`
- Driver: 595.71.05
- Torch: `2.10.0+cu129`; CUDA runtime: 12.9
- TileLang: `0.1.11+cu129.gitafcebed1`
- vLLM: `0.19.1`
- TileFoundry: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`

The pinned vLLM CUTLASS module exposes FP8, FP4 and W4A8/FP8 expert classes,
but no matching BF16 entry point. It is not presented as an equivalent
baseline. The image lacks `cupti-python`, and dependencies were not changed;
all final comparison rows explicitly use the benchmark's CUDA-events fallback.

## Plan audit and blind phase

The draft plan needed three corrections before implementation:

1. General vector `IndexSelect` evaluates and analyzes, but cannot lower to
   TIR; the plan therefore had to include a handwritten semantic twin and a
   structured lowering gap instead of assuming generated runtime code.
2. The benchmark's stale CUTLASS imports were not a same-contract BF16
   baseline and had to be removed. The workload, not the benchmark, now owns
   the fallback oracle.
3. The pinned image cannot run the new native CUPTI harness. The honest option
   available without replacing dependencies is one explicit CUDA-events method
   for candidate, incumbent and vLLM.

The intended blind phase was compromised at
`2026-08-12T00:09:37+08:00`: an incorrectly scoped negative glob searched an
incumbent grouped-GEMM file. It printed only matching file paths and line
numbers, not bodies or configurations. No incumbent idea was used in the first
candidate, but this was still a literal brief violation. Incumbent bodies were
first opened only after `first-candidate.md` was saved at
`2026-08-12T00:37:16+08:00`.

The first candidate is `first_candidate_direct.py`; it uses two TileLang
launches, supports the requested BF16/SILU/single-GPU semantics, and passed a
small repeated-ID/scaling-0.75 check with max absolute error
`1.1920928955078125e-07`. Its first production decode samples were
`[953.082336, 953.083923, 953.032410]` ms. The lack of expert grouping causes
each route to reread selected expert weights, so it was rejected before any
production dispatch change.

## Authored HIR and TileFoundry

- H200 semantic graph: `authored_hir.py`
- CPU evaluator twin: `authored_hir_twin.py`
- Check inputs/checkpoint: `hidden.pt`, `topk_weights.pt`, `topk_ids.pt`,
  `model.safetensors`
- Raw analysis: `authored-hir-analyze.json`
- Raw check: `authored-hir-check.json`
- CTA advice: `authored-hir-schedule.log`
- Structured gaps and reproducer: `tilefoundry-gaps.md`,
  `index_select_lowering_repro.py`, `index-select-lowering-error.log`

The exact HIR graph passed type inference and H200 analysis: 99,072 BF16
FLOPs, 1,792 FP32 FLOPs, 209,504 bytes read, 110,080 bytes written, and a 67 ns
small-shape memory roofline. The evaluator twin passed `allclose` at
`atol=rtol=0.01` with max violation 0 and passed NaN/Inf checks. The CPU alias
is required only because the host evaluator could not create a cuBLAS handle
while the persistent container owned the GPU; H200 analysis/scheduling still
uses `nvidia.h200_sxm`.

## Original CUDA 12.9 final performance

Raw report: `evidence/profile_run.log`. JUnit and stdout:
`evidence/benchmark-full.xml` and `evidence/benchmark-full.log`.

All rows are BF16, `K=8`, `H=7168`, `F=2048`. Latency and p10-p90 are ms.
`inc speedup` is incumbent latency divided by candidate latency.

| workload | candidate | p10-p90 | incumbent | p10-p90 | vLLM | p10-p90 | inc speedup | vLLM/candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-235b-decode | 3.0369 | 3.0308-3.0493 | 3.1552 | 3.1469-3.1737 | 3.0022 | 2.9848-3.0260 | 1.0390x | 0.9886x |
| qwen3-235b-prefill | 6.0420 | 5.7108-6.3969 | 5.7479 | 5.7391-5.7678 | 7.3834 | 7.0425-8.5899 | 0.9513x | 1.2220x |
| deepseek-v3-decode | 5.7930 | 5.7737-5.8215 | 6.0943 | 6.0854-6.1130 | 5.6403 | 5.6260-5.6723 | 1.0520x | 0.9736x |
| deepseek-v3-prefill | 8.1298 | 7.8985-8.2797 | 8.1776 | 8.1000-8.3025 | 8.6845 | 8.6490-8.7859 | 1.0059x | 1.0682x |
| geometric mean | 5.4219 | - | 5.4830 | - | 5.7403 | - | 1.0113x | 1.0587x |

The candidate is faster than incumbent on both decode rows and the
256-expert prefill row, with a 1.13% geometric-mean improvement, but it is
4.87% slower on 128-expert prefill. It has a lower geometric mean than vLLM,
but both decode rows are slower; their p10 is above vLLM's p90. Therefore the
brief's every-row SOTA condition is not met.

## Profiling and tuning

NCU cannot access H200 counters under the host policy and returns
`ERR_NVGPUCTRPERM`; raw output is `evidence/ncu-permission.log`. No occupancy,
DRAM or tensor-core percentage is claimed. PyTorch CUDA profiler traces and
tables are preserved under `evidence/profile-*.json` and `.log`.

- 256-expert decode candidate: five TileLang launches. Fused gate/up takes
  3.725 ms (64.9% of non-overlapped CUDA operator time), down GEMM 1.961 ms
  (34.2%), and scan/gather/unpermute total 49.8 us. This is the sparse route
  shape selected for `BM64/stage2`.
- 128-expert prefill candidate: fused gate/up 3.697 ms, down GEMM 1.861 ms,
  and routing/finalize 392.3 us. The incumbent's two GEMMs total 5.213 ms and
  separate activation is only 88.7 us, explaining the candidate's roughly
  0.26 ms dense-route regression.

Measured tuning budget and decisions:

- Removing the fused gate/up output zero-fill changed qwen decode from 3.0448
  to 3.0349 ms in the exploratory harness. Every valid output element is
  overwritten; empty experts own no output rows.
- For 256-expert decode, gate/up `BM64/stage2` measured 3.7667 ms versus
  3.8561 ms for `BM128/stage3`; full-op A/B was 5.8068 versus 5.8863 ms, while
  vLLM moved only 5.6412 to 5.6393 ms. The sparse configuration was retained.
- For 128-expert decode, `BM128/stage3` was the best viable cooperative
  candidate (1.9274 ms gate/up); `BM64/stage2` was 1.9380 ms. `BM64/stage3`
  exceeded the measured shared-memory limit and was correctly pruned by the
  existing autotune calculation.
- Expert-group swizzles of 4, 8 and 16 increased gate/up latency; all were
  reverted. A first attempt also placed a macro in the wrong template scope
  and failed compilation before measurement; it was corrected, measured, and
  then removed.

Ideas adopted after inspecting the incumbent: expert grouping/counting-sort
is necessary to reuse expert weights, and the existing grouped kernel's
full-write invariant justifies replacing `zeros` with `empty`. The rejected
direct candidate adopted no incumbent idea. The shipped code extends the
existing fused kernel; it is not relabelled as TileFoundry-generated.

## Correctness and blast radius

All TileOPs Python/GPU commands ran through the persistent container. Raw logs
and JUnit files are in `evidence/`.

```text
python -m pytest -q tests/ops/test_fused_moe_experts.py -m smoke
  33 passed, 1 deselected
python -m pytest -q tests/kernels/test_moe_grouped_gemm_3wg_fused_act.py
  9 passed
python -m pytest -q tests/test_validate_manifest.py
  122 passed
python -m pytest -q benchmarks/tests
  22 passed
python -m pytest -q tests/test_workload_placement.py
  2 passed
python -m pytest -q -s benchmarks/ops/bench_fused_moe_experts.py
  4 passed
python -m ruff check <four changed Python files>
  All checks passed
```

The kernel signature and return shape are unchanged. Its direct kernel tests,
the public operator's FP16/BF16, fallback, EP and activation smoke paths, the
four benchmark callers, manifest validation and workload placement were run.
The shared workload method has only the benchmark and the two existing
default-SiLU correctness paths as callers. The benchmark asserts every built
candidate kernel is `MoeGroupedGemmPersistent3WGFusedActKernel`, proving no
measured fallback. `test_node_delta.py` reports 33 to 34 nodes (+1, 3.0%); the
single node covers the new sparse configuration with empty experts and a
partial M tile. The first raw node-delta attempt is also retained: it exposed
that the container's worktree gitdir is not mounted, then a self-contained git
bundle supplied the same base commit for the corrected run.

GitHub CI passed pre-commit, benchmark contracts, compile contracts, packaging,
gitleaks, actionlint, security policy and GPU smoke. The uploaded GPU artifact
is preserved under `evidence/ci/`: the repository smoke shard reports 2,915
passed and 14 skipped with zero failures, and the diff-scoped full run reports
33 passed with zero failures. `evidence/ci/gpu_smoke_results.xml` and
`gpu_smoke_full_results.xml` are the CI JUnit files; `evidence/pr-status.json`
and `evidence/pr-checks.log` preserve the final PR state.

## Original CUDA 12.9 residual risk

- CUDA-events fallback has wider attribution limits than native CUPTI, though
  every comparison implementation used it consistently in one process.
- Hardware-counter profiling is unavailable under current permissions.
- The sparse threshold is H200/SM90 evidence-driven. Other architectures
  cannot build this kernel because its declared support remains SM90 only.
- Random manifest routing covers realistic average load but not every extreme
  production skew; direct skew/empty/tail kernel tests cover correctness.
- The opt-in fused path remains slower than incumbent on the 128-expert prefill
  row and slower than vLLM on both decode rows. The public default remains the
  unfused pipeline; changing that default would be a separate human contract
  decision and was not done here.
