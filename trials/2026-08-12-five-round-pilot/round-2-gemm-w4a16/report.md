# Round 2 report: GemmW4A16Op

## CUDA 13.2 retest addendum (2026-08-12)

The original PR commit was retested in the admitted CUDA 13.2 runner image
`ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev` at digest
`sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`.
The stack was Python 3.12.13, driver 595.71.05, CUDA 13.2, PyTorch
2.13.0+cu132, TileLang 0.1.11+cu132.gitafcebed1, and vLLM 0.27.1 on an
NVIDIA H200.

All six manifest correctness rows passed. The targeted W4A16 suite passed 5
tests, the Marlin conversion suite passed both reduction modes, the complete
benchmark contract suite passed 24 tests, and manifest validation passed 122
tests. The official performance run completed all four primary rows and every
comparator in one process with native CUPTI attribution.

CUDA 13.2 native-CUPTI medians in ms:

| workload | candidate | incumbent | Marlin FP16 | Marlin FP32 | PyTorch oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode-l2-resident-ish | 0.0538395 | 0.0844635 | 0.020160 | 0.020064 | 0.045824 |
| decode-hbm-streaming-threshold | 0.101120 | 0.168576 | 0.0352635 | 0.035328 | 0.073760 |
| decode-non-power2-low-cta | 0.122688 | 0.2040485 | 0.0379845 | 0.037952 | 0.0845445 |
| decode-long-k-pressure | 0.463393 | 0.809922 | 0.124512 | 0.124768 | 0.322464 |

The candidate remains an **improvement without SOTA**: per-row incumbent
speedups are 1.5688x, 1.6671x, 1.6631x, and 1.7478x; geometric-mean speedup is
1.6605x. It remains 3.1019x slower than the fastest passing Marlin mode by
geometric mean. Candidate p10-to-p90 widths are 2.318%, 1.266%, 1.356%, and
0.642%. This table supersedes the pre-rebase CUDA 13.2 table because rebasing
adopted the call-window CUPTI attribution and interleaved comparison framework.

The CUDA 13.2 GPU Smoke rerun exposed two pre-existing generic FP16 GEMV test
tolerance failures. The same two failures reproduced locally and do not touch
the W4A16 implementation. TileOPs PR #1895 independently diagnosed the CUDA
13.2 reduction-order effect and merged a GEMV-only, K-scaled tolerance fix to
`main`. Rebasing this PR onto that commit is therefore the scoped resolution;
the W4A16 runtime implementation needs no compatibility change.

The PR was rebased onto `main` commit
`994f43d374cbc1d77a08e207ab8831402b87a376`; its rewritten head is
`2f0f1eb62f682208757ad29a99ee91f5a8a5e8a0`. The benchmark conflict was
resolved by retaining the common-input Marlin correctness gate while adopting
the upstream interleaved `ManifestBenchmark.compare` path. Since that rebase
changed CUPTI attribution and comparison behavior, the full primary
performance distribution above was refreshed rather than carrying forward the
pre-rebase medians.

Post-rebase local evidence: complete GEMM test file 39 passed; W4A16 plus
Marlin adapter 7 passed; manifest W4A16 benchmark 6 passed; benchmark contract
tests 20 passed; manifest validator 122 passed; static checks passed. Remote CI
is fully green at the rewritten head. Its changed-file GPU JUnit reports 39
passed, and the global GPU smoke JUnit reports 2931 passed and 14 skipped. The
PR is `MERGEABLE`; `mergeStateStatus=BLOCKED` only because the requested
`tileops-review` team has not yet supplied the required non-author approval.
There are no reviews, review comments, or review threads. The PR was not
merged.

## Outcome

Classification: **improvement without SOTA**.

The delivered M=1 TileLang decode kernel is faster than the base-commit
TileOPs incumbent on all four primary rows: 1.601x, 1.687x, 1.675x, and
1.760x, with a 1.680x geometric-mean speedup. It is not external SOTA: its
geometric-mean latency is 3.29x slower than the fastest correctness-passing
Marlin mode selected independently on each row. The result remains useful as
a reviewable improvement to TileOPs, so it was submitted without an SOTA
claim.

## Delivery and environment

- Branch: `perf/tileops-r2-gemm-w4a16`
- Base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`
- Commit: `b94331bcbae7b15201a3e31367e6d5588809626f`
- PR: <https://github.com/tile-ai/TileOPs/pull/1887>
- PR status at report finalization: open, all CI green,
  `mergeable=MERGEABLE`. `mergeStateStatus=BLOCKED` only because `main`
  requires one non-author approving review; team `tileops-review` is requested
  and there are no review comments. The PR was not merged.
- Container: `tileops-db8dd17a394e`, persistent GPU 0, NVIDIA H200 (SM90)
- Image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`, digest
  `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`
- Python 3.12.13; CUDA 12.9; PyTorch 2.10.0+cu129; TileLang
  0.1.11+cu129.gitafcebed1; vLLM 0.19.1; cupti-python 12.8.0
- TileFoundry: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`, invoked only through
  `$HOST_HOME/TileFoundry/.venv/bin/tilefoundry`; its checkout was not
  modified.

All artifact names below are relative to
`$TRIAL_SOURCE/round-2-gemm-w4a16/` unless an absolute
path is shown.

## Blind phase and authored HIR

The incumbent remained unread until the first candidate had passed a real GPU
compile-smoke row and `decode-l2-resident-ish`, and `first-candidate.md` was
written. The blind phase ended at exactly `2026-08-12T02:04:48+08:00`.
`blind-read-audit.md` records every blind-phase source-read command.

- Exact authored graph: `authored_hir.py`. It preserves low-nibble-even/high-
  nibble-odd interleave, group-128 metadata expansion, FP32 affine
  dequantization, the FP16 cast before matmul, transpose, and matmul.
- Exact import result: `hir-import.log`; current TileFoundry rejects the two
  `u8` tensor types before parsing the function body.
- Dynamic i32-carrier reproducer: `authored_hir_i32_dynamic_repro.py` and
  `hir-i32-dynamic-repro.log`; `RepeatInterleave` shape inference fails on the
  symbolic `(K // 128) * 128` extent even when K is CLI-bound.
- Static i32-carrier semantic workaround:
  `authored_hir_i32_workaround.py`. Its evaluator is bitwise equal at
  `(64,64,128)` (`hir-i32-evaluator-smoke.log`), `tilefoundry check` passes at
  zero tolerance (`hir-i32-check-smoke-2.json`), and analyze succeeds
  (`hir-i32-analyze-smoke.json`).
- Schedule result: `hir-i32-schedule-smoke-3.json`; H200 target costing has no
  dense i32 peak and refuses to schedule the graph.
- Structured gap report: `tilefoundry-gaps.md`.

The immutable first candidate used `block_n=8, threads=128`. It passed the
compile-smoke row and the full `(1,8192,8192)` numerical oracle with
`max_abs=0.03125`, `max_rel=0.00508044`. Three independent native-CUPTI runs
had medians 0.064705, 0.064800, and 0.064705 ms. See `first-candidate.md`,
`first-smoke.xml`, `first-smoke-3.log`, and
`first-candidate-decode-l2.log`. The first candidate file was named
`src/tileops/kernels/gemm_w4a16_r2.py` at snapshot time; after the blind phase
it was renamed to the delivered `gemm_w4a16_decode.py`.

## Delivered design and blast radius

`GemmW4A16Kernel` delegates M=1 to the new fused decode class and retains the
base implementation unchanged for M>1. The public Op signature, validation,
output shape/dtype, and cache key `(M,N,K,dtype,group_size)` are unchanged.
Repository caller search found the kernel constructor is reached through
`GemmW4A16Op`; the benchmark imports the new class only for evidence.

Each CTA handles 16 output rows with 128 threads and iterates K in group-128
steps. It loads one 128-element FP16 activation tile, reads each packed byte
once, computes low/high dequantized FP16 values in separate fragments, and
accumulates products in FP32. It emits only `[M,N]`; it never creates an
`[N,K]` runtime allocation.

The Marlin adapter now reconstructs the same logical q tensor from TileOPs
little-nibble-first storage, then uses vLLM 0.19.1's installed official
`get_weight_perm`, `marlin_weights`, `marlin_permute_scales`, and
`marlin_zero_points` helpers. Repacking, scale conversion required by Marlin,
zero-point conversion, and workspace allocation are all outside timing. Both
reduction modes are checked against the same FP16-rounded oracle before they
are eligible as baselines.

Blast-radius coverage includes all 39 GEMM op nodes, all 24 benchmark contract
nodes, all six W4A16 manifest rows, both Marlin modes, manifest validation, and
the new packing/group-boundary regression. No shared function signature or
return shape changed.

## Correctness evidence

Required and expanded runs:

| Scope | Result | JUnit / raw artifact |
| --- | --- | --- |
| `pytest -q tests/ops/test_gemm.py -k 'W4A16 or w4a16'` | 5 passed, 33 deselected | `w4a16-targeted.xml`, `w4a16-targeted.log` |
| six exact manifest rows plus dispatch assertions | 6 passed | `manifest-correctness.xml`, `manifest-correctness.log`, `test_manifest_correctness.py` |
| `pytest -q tests/test_validate_manifest.py` | 122 passed | `validate-manifest.xml`, `validate-manifest.log` |
| complete `tests/ops/test_gemm.py`, four compiler-process shards | 15 + 6 + 6 + 12 = 39 passed | `gemm-shard-*.xml`, `gemm-shard-*.log` |
| benchmark base/boundaries/Marlin tests | 12 + 2 + 2 passed | `test_benchmark_base.xml`, `test_benchmark_boundaries.xml`, `test_gemm_w4a16_marlin.xml` and matching logs |
| benchmark runner process-control nodes | 8 individually isolated, all passed | `run-bench-*.xml`; verbose logs retained for the two diagnostic reruns |
| `pytest -q -s benchmarks/ops/bench_gemm.py -k w4a16` | 6 passed, 28 deselected | `bench-gemm-w4a16-s.xml`, `bench-gemm-w4a16-s.log`, `bench-gemm-w4a16-s-profile_run.log` |
| focused packing boundary | 1 passed | `nibble-boundary.xml`, `nibble-boundary.log` |

Running the complete GEMM file in one persistent pytest process exited without
a summary/JUnit after accumulating 27 compiled variants; the same occurred
when the benchmark process-control tests that intentionally create crashes and
hangs were combined. These incomplete attempts are retained as
`gemm-full.log`, `gemm-full-retry.log`, and `benchmarks-tests.log` and are not
claimed as passes. Every collected node was then run in a smaller isolated
process and passed with JUnit evidence as listed above.

Static checks: Ruff, shipped-source reference lint, compileall, and
`git diff --check` passed (`ruff.log`, `shipped-refs-lint.log`). Test-node delta
was +1 in the existing GEMM test file and +2 in a new Marlin adapter test file;
no node was removed (`test-node-delta-explicit.log`).

PR CI is green at commit `b94331b`: pre-commit, gitleaks, actionlint,
packaging-check, compile-contract-gate, benchmark-contract-tests, and GPU smoke
all passed. GPU smoke ran the 39-node changed-file set and then reported
`2931 passed, 14 skipped, 1133 deselected`; its uploaded logs and JUnit are in
`ci-gpu-smoke-artifact/`, including `gpu_smoke_results.xml` and
`gpu_smoke_full_results.xml`. The complete job log is `ci-gpu-smoke-job.log`.
Benchmark contract CI reported `20 passed, 4 skipped` in
`ci-benchmark-contract-job.log`; the four GPU-dependent nodes pass locally in
the JUnit artifacts above.

## Final benchmark

Method: one persistent container/process/GPU; exact same logical tensors for
candidate, base incumbent, both Marlin modes, and oracle; native CUPTI only;
25 ms warmup and 100 ms repeat policy; L2 reset and shifted pointers before
samples; conversion, compilation, tuning, workspace allocation, and oracle
construction excluded. Forward and reverse execution passes were combined.
Every raw sample and both pass medians are in `final-benchmark.log`; the exact
driver is `final_benchmark.py`.

Latency in ms (`median [p10,p90]`):

| workload | candidate | base incumbent | Marlin FP16 | Marlin FP32 | PyTorch oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode-l2-resident-ish | 0.061313 [0.060704,0.061921] | 0.098176 [0.097824,0.098688] | 0.021872 [0.021569,0.022496] | **0.021776** [0.021472,0.022432] | 0.048224 [0.047392,0.313217] |
| decode-hbm-streaming-threshold | 0.116704 [0.116065,0.117504] | 0.196881 [0.196161,0.197601] | 0.038097 [0.037664,0.038848] | **0.037984** [0.037536,0.038784] | 0.083633 [0.082464,0.323233] |
| decode-non-power2-low-cta | 0.142273 [0.141568,0.143265] | 0.238305 [0.237537,0.239521] | 0.040768 [0.040225,0.041632] | **0.040753** [0.040193,0.041663] | 0.089280 [0.088097,0.319553] |
| decode-long-k-pressure | 0.544067 [0.542242,0.546370] | 0.957620 [0.956195,0.960835] | **0.139857** [0.138944,0.141313] | 0.139937 [0.138913,0.141217] | 0.363105 [0.360738,0.365858] |

The candidate p10-p90 widths are 1.98%, 1.23%, 1.19%, and 0.76% of the
median. Forward/reverse pass medians are also recorded in the raw log and are
well inside those intervals.

| metric | candidate | base incumbent | fastest passing Marlin / row | oracle context |
| --- | ---: | ---: | ---: | ---: |
| geometric-mean latency (ms) | 0.153410 | 0.257711 | 0.046597 | 0.106932 |
| candidate speedup | 1.000x | 1.680x | 0.304x | not a competing baseline |

Per-row candidate speedups over base are 1.601x, 1.687x, 1.675x, and 1.760x.
The candidate is respectively 2.816x, 3.072x, 3.491x, and 3.890x slower than
the fastest equivalent Marlin mode. The gap is far larger than measured
noise on every row, establishing the non-SOTA classification.

Maximum absolute / relative error versus the shared oracle:

| workload | candidate | base incumbent | Marlin FP16 | Marlin FP32 |
| --- | ---: | ---: | ---: | ---: |
| decode-l2-resident-ish | 0.03125 / 0.00508 | 0.0625 / 0.47822 | 0.0625 / 0.49909 | 0.0625 / 0.48276 |
| decode-hbm-streaming-threshold | 0.0625 / 0.01462 | 0.0625 / 1.27284 | 0.0625 / 1.47625 | 0.0625 / 1.29354 |
| decode-non-power2-low-cta | 0.0625 / 0.06202 | 0.0625 / 0.92802 | 0.125 / 3.78848 | 0.0625 / 0.92470 |
| decode-long-k-pressure | 0.25 / 0.11550 | 0.125 / 3.25691 | 0.25 / 2.63300 | 0.125 / 3.27348 |

Large relative maxima occur at oracle values near zero. Every implementation
listed passed the unchanged `atol=0.07, rtol=0.05` elementwise criterion; no
tolerance was loosened.

## Tuning and profiler evidence

The controlled sweep evaluated six configurations, the Cartesian product of
`block_n={4,8,16}` and `threads={128,256}`, on every primary row: 24
shape/config evaluations. All were correctness checked and timed with CUPTI.
The sweep took 113.612 seconds, or 0.0316 H200 GPU-hours/compile-hours. The
single configuration winning every row was `block_n=16, threads=128`, selected
without dropping a row. Raw samples are in `tuning.log`; driver:
`tune_candidate.py`.

Against the blind first candidate (`8,128`), `16,128` reduced latency by about
5.4% on L2, 5.9% at the HBM threshold, 3.9% on the non-power-of-two row, and
11.0% on long K. Moving to 256 threads regressed every tested block size. These
ablations motivated increasing output-row reuse per CTA while retaining 128
threads.

PyTorch profiler traces cover `decode-l2-resident-ish` and
`decode-long-k-pressure`: `decode-l2-resident-ish-trace.json`,
`decode-long-k-pressure-trace.json`, and `profile-candidate.log`. On L2, three
custom-op calls produced exactly three `main_kernel` and three
`cuLaunchKernel` events. On long K, three custom-op calls produced three
`cuLaunchKernel` events; the aggregate table includes one warmup/schedule-bleed
`main_kernel` event, so launch attribution is based on the custom-op and CUDA
launch rows, not that aggregate count. Three calls allocate only three 16 KiB
outputs (48 KiB total), with no other CUDA allocation.

Generated sources `candidate-l2-generated.cu` and
`candidate-long-k-generated.cu` each contain one `main_kernel`, a 128-element
FP16 activation shared array (256 bytes), and no `cudaMalloc`, full `[N,K]`
buffer, or second kernel. TileLang exposes `_primary_resource_usage`, but it
returned `None` for both shapes (`resource-usage.log`); register/occupancy
numbers are therefore not inferred. NCU counters were unavailable by fixed
preflight (`ERR_NVGPUCTRPERM`), so the evidence uses CUPTI, profiler events,
generated code, and the controlled ablation as required.

## Failed approaches and attribution

- Exact TileFoundry HIR failed on missing `u8`; a dynamic i32 workaround then
  failed `RepeatInterleave` symbolic shape inference. The static i32 workaround
  checked semantics but could not schedule because H200 has no dense i32 peak.
- The first TileLang runtime attempt reused one fragment for low/high products;
  TileLang's data-race verifier rejected inconsistent index expressions
  (`first-smoke.log`). Separating low/high fragments preserved the graph and
  passed verification.
- A vectorized strided activation load was rejected because TileLang could not
  prove unit source scale (`first-smoke-2.log`); loading the contiguous
  group-128 activation slice resolved it.
- A guessed `pack_cols + awq_marlin_repack` adapter executed but failed common-
  input equivalence (`marlin-common-input-check.log`). It was discarded and
  replaced with vLLM's official Marlin layout/permutation helpers; both modes
  then passed (`marlin-common-input-check-2.log`).
- Smaller `block_n` and all 256-thread configurations were correct but slower;
  no slow row was removed.

Incumbent ideas adopted into the new M=1 algorithm: **none**. The authored HIR
and first correct runtime twin predate any incumbent read. After the blind
boundary, the existing incumbent was retained verbatim as the M>1 path; this
reuse is explicit and is not attributed to TileFoundry.

## Residual risk

- Marlin remains 2.82-3.89x faster. Closing that gap likely needs substantially
  better packed-weight/vector memory use and reduction strategy; the current
  profiler evidence is insufficient to make a narrower claim without hardware
  counters.
- TileLang's resource API returned no register/occupancy report, and NCU access
  is unavailable. Extremely different decode shapes beyond the manifest may
  expose resource behavior not measured here.
- The benchmark depends on helpers from vLLM's installed
  `marlin_utils_test` module because those are the official layout constructors
  in the pinned version. This is benchmark-only and correctness guarded, but it
  is not a stable vLLM public API across upgrades.
- Consolidated long-lived pytest processes exited in the local persistent
  container after accumulated compiler/native-process tests. All nodes passed
  in isolated local processes with JUnit. The clean PR CI subsequently passed
  the benchmark contract suite in one process and the 39-node GEMM changed-file
  set, reducing this to a persistent-local-container observation rather than a
  merge blocker.
