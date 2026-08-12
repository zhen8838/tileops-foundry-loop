# Round 5 FFTC2COp report

## Result

- Classification: **improvement without SOTA**.
- Branch: `perf/tileops-r5-fft-c2c`.
- Base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`.
- Commit: `bf8d0599325209319c365a3488338ec6905691d6`.
- PR: https://github.com/tile-ai/TileOPs/pull/1892 (open; all CI green; GitHub
  `mergeable=MERGEABLE`; awaiting the repository-required human review).

The retained change removes both real/imaginary input-copy kernels from the
default contiguous public call, feeds a metadata-only interleaved real view to
TileLang, reuses the precomputed twiddle LUT in shared-memory stages, and sets
the measured N=4096 default to 256 threads/block work items. It preserves the
public op, split-input kernel, custom `kernel_map`, validation, cache, output,
and non-contiguous/conjugate behavior. It is useful independently of SOTA: all
three primary rows improve over the fixed base, with a 10.65x geometric-mean
speedup, but all remain slower than cuFFT.

## Plan audit

The draft brief was checked against the checkout before implementation and
corrected as follows:

1. `load_manifest()` returns the merged op mapping directly; the structural
   lookup is `load_manifest()["FFTC2COp"]`, with exactly the three stated rows.
2. The existing public wrapper launched two `real.contiguous()` /
   `imag.contiguous()` copy kernels. These are part of the contract timing;
   kernel-only timing would not answer the brief.
3. The repository benchmark already provides fail-closed native CUPTI,
   complete-sequence attribution, L2 reset, and shifted input addresses. The
   final harness extended this path instead of changing the timing contract.
4. Existing op tests stopped at N=1024 and omitted N=1/N=2, every exact
   manifest row, and non-contiguous views. Nine focused nodes were added.
5. The HIR pseudocode was illustrative: current TileFoundry uses `tf.concat`,
   `tf.matmul`, tuple returns, `@module`, and explicit target/topology objects.
   The authored graph was adapted to those actual surfaces without changing
   orientation, precision claims, or arithmetic.
6. After the blind seal, incumbent inspection showed that its shared-memory
   phase still called runtime `T.cos/T.sin`, despite its surrounding
   precomputed-LUT description. The optimization plan therefore added LUT use
   to those stages and corrected the stale documentation. The class also
   overclaimed cuFFT-level performance and misstated the fused stage count by
   `+1`; both statements were corrected to match code and measurements.

## Environment

- Image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`, brief-pinned
  digest `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`.
- GPU: NVIDIA H200; driver 595.71.05; observed 1500 MHz graphics and 3201 MHz
  memory clocks.
- PyTorch 2.10.0+cu129; CUDA 12.9.
- TileLang `0.1.11+cu129.gitafcebed1`; `cupti-python` 12.8.0.
- TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
- Every TileOPs Python/GPU command ran through the one persistent
  `tileops-container.sh` container. No dependency was installed, upgraded, or
  replaced. The image lacks `pre-commit`; GitHub CI ran the authoritative hook.

## Blind phase and authored HIR

The blind phase ended at `2026-08-12T05:57:26.225141650+08:00`. Before that
timestamp, reads were limited to the brief-authorized public contract,
manifest, workload, op wrapper, tests, benchmark harness, preflight evidence,
TileFoundry specifications/tutorials/implementation surfaces needed for syntax,
and unrelated TileLang conventions. Neither `src/tileops/kernels/fft.py`, a
generated/cache copy, nor `tests/kernels/test_fft_output_layout.py` was opened or
searched. The ordered read audit is in `first-candidate.md`.

`authored_hir.py:FFTPairF32` defines a typed f32 real-pair complex butterfly and
direct DFT with explicit `W[t,k]` twiddles and natural-frequency output:

- `check`, evaluator oracle, `analyze`, `schedule`, canonical source round-trip,
  and six CLI cases complete. The evaluator's largest error is
  `1.1920928955078125e-7`.
- Analysis reports 272 f32 FLOPs, 512 bytes read, and 192 bytes written. The
  six-op H200 schedule is `FEASIBLE_NOT_PROVEN`, objective 6 ns; this is advice,
  not measured runtime.
- Direct `complex64`, `complex128`, and paired `f64` attempts fail with exact
  `DType: unknown value ...` diagnostics. The f32 graph is not presented as a
  complex128 description.
- Authored HIR SHA256:
  `14b7d96cd162a0e5ae8eeddd715fff2d222feeebd62f9e6556554519ad0b9056`.

The sealed runtime twin is an independent radix-2 TileLang FFT with a cached
shape-only bit-reversal permutation, twelve explicit butterfly stages, cached
twiddles, and direct interleaved output. It calls neither incumbent nor cuFFT
and caches no input-dependent value. N=8/B2 passes a literal direct DFT with
max/mean complex error `5.663382e-6 / 2.021699e-6`; N=4096/B1 c64 passes
`torch.fft.fft` with `4.533716e-5 / 1.112630e-5`. Its three native-CUPTI trial
medians are 0.0688645, 0.0727840, and 0.0700160 ms.

Seal:

```text
first-candidate.md SHA256
20e8a9e92230794ebd5ac5ce8b4307866039653b51221232347e286eef9455de
```

Primary blind artifacts are `first-candidate.md`, `first-candidate.sha256`,
`artifacts/blind-first-candidate.{json,xml}`, the three
`authored-hir-{check,analyze,schedule}.json` files,
`authored-hir-oracle.{json,xml}`, and
`evidence-sources/fft_blind_candidate.generated.cu`.

## Final correctness

Candidate, reconstructed fixed-base incumbent, and warmed cuFFT used the exact
same input in one process. The base source SHA256 is
`8c64d6f3f0e5f0e8ba08e7459e0951c325daf0a6d64f33d27658eaf0b961c533`;
the harness changes only its custom-op symbol so both versions can coexist.

| Row | Candidate max / mean complex abs | Base max / mean complex abs | Candidate assert_close |
| --- | ---: | ---: | --- |
| 4K / B1 / c64 | 4.264961e-5 / 1.090792e-5 | 7.864200e-5 / 1.866139e-5 | pass at 1e-4/1e-4 |
| 4K / B64 / c64 | 5.554283e-5 / 1.082392e-5 | 1.078959e-4 / 1.846451e-5 | pass at 1e-4/1e-4 |
| 4K / B64 / c128 | 1.994589e-13 / 3.616139e-14 | 2.089769e-13 / 3.576006e-14 | pass at 1e-8/1e-8 |

Tests additionally cover exact shape/dtype, natural ordering, default forward
normalization, zero/single-stage transforms, leading-dimension flattening,
strided and unresolved-conjugate inputs, output layout, and the existing
four-tensor custom-kernel contract. The N=8 blind fixture supplies the
independent direct-DFT oracle.

## Final performance

Protocol: one persistent container, process, H200, input set, precision
contract, and native-CUPTI harness. All implementations were warmed, including
three distinct cuFFT plan-cache entries. Compilation, tuning, LUT creation, and
plan creation were excluded. Each implementation occupied each position in a
three-order rotation; every trial retained 200 raw samples with L2 reset/input
shifting. The full public call, copies/views, output materialization, and
synchronization remained inside timing. No timing fallback occurred.

Raw trial medians (ms):

| Row | Candidate | Base incumbent | cuFFT | trial-median span C / B / cuFFT |
| --- | --- | --- | --- | ---: |
| 4K / B1 / c64 | 0.0072165, 0.0089920, 0.0071360 | 0.1019360, 0.1155360, 0.1851845 | 0.0051520, 0.0051680, 0.0051840 | 25.72% / 72.05% / 0.62% |
| 4K / B64 / c64 | 0.0120960, 0.0120960, 0.0120960 | 0.1165920, 0.1084005, 0.1077120 | 0.0056000, 0.0056320, 0.0056320 | 0.00% / 8.19% / 0.57% |
| 4K / B64 / c128 | 0.0152960, 0.0152960, 0.0152960 | 0.1268965, 0.1288805, 0.1292965 | 0.0084480, 0.0084800, 0.0084800 | 0.00% / 1.86% / 0.38% |

Reported values are medians of the three trial medians:

| Row | Candidate ms | Base ms | cuFFT ms | Base / candidate | Candidate / cuFFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K / B1 / c64 | 0.0072165 | 0.1155360 | 0.0051680 | 16.0100x | 1.3964x slower |
| 4K / B64 / c64 | 0.0120960 | 0.1084005 | 0.0056320 | 8.9617x | 2.1477x slower |
| 4K / B64 / c128 | 0.0152960 | 0.1288805 | 0.0084800 | 8.4258x | 1.8038x slower |
| geometric mean | 0.0110116 | 0.1173038 | 0.0062728 | 10.6528x | 1.7554x slower |

The B1 base and candidate contain isolated slow samples/modes, but the result is
directionally robust: even the candidate's slow trial remains far faster than
the base's fastest trial, while even its fastest trial is slower than cuFFT's
slowest trial. SOTA requires no slower row and a lower geometric mean; neither
condition holds. The honest classification is therefore **improvement without
SOTA**.

Raw evidence is `worktree-artifacts/final/final-evidence.{json,xml}`; SHA256 is
`1ec810b45dbb084ddf38213457d01c22fe583ff3a8a87969878d8fdbdae78341`
for JSON and
`43a8c6b29f22a2ee92aedba34bf03ec2da874ba1b0c1a2babf68a06feae65a9a`
for the zero-failure JUnit XML. The final official benchmark report is
`worktree-artifacts/validation/post-cleanup-profile_run.log`.

## Profiler, copies, memory, and bottleneck

Native CUPTI launch traces prove candidate dispatch without cuFFT/incumbent:

- Candidate: exactly two `_fft_lut_main*` kernels and no input-copy kernel.
  Representative kernel durations are 2.72-2.85 + 1.47-1.54 us (B1 c64),
  7.58-7.74 + 2.05-2.21 us (B64 c64), and 10.24-10.43 + 2.94-3.17 us
  (B64 c128).
- Base: two PyTorch `direct_copy_kernel_cuda` launches followed by the same two
  FFT phases. Representative copies are 1.38-1.47 us each (B1), 2.18-2.40 us
  each (B64 c64), and 2.40-2.78 us each (B64 c128).
- cuFFT: one `vector_fft<4096...>` kernel, 4.16-4.19 us, 4.32-4.39 us, and
  6.94-7.23 us across the three rows.

| Row | Candidate output / temporary / peak increment | Base temporary / peak increment | cuFFT temporary |
| --- | ---: | ---: | ---: |
| B1 c64 | 32 / 32 / 64 KiB | 64 / 96 KiB | 0 KiB |
| B64 c64 | 2048 / 2048 / 4096 KiB | 4096 / 6144 KiB | 0 KiB |
| B64 c128 | 4096 / 4096 / 8192 KiB | 8192 / 12288 KiB | 0 KiB |

Generated CUDA shows `__launch_bounds__(256,1)`, 4 KiB dynamic shared memory
per c64 CTA and 8 KiB per c128 CTA. The radix-8 tail declares eight local
complex pairs; exact post-allocation register counts are unavailable because
NCU is blocked by `ERR_NVGPUCTRPERM`, so no register-count claim is made.
Separate PyTorch profiler evidence has 12 candidate, 23 base, and 13 cuFFT
events per row, with all nine correctness checks passing; traces and parsed
data are under `worktree-artifacts/final/`.

The concrete remaining bottleneck is the two-phase global-memory algorithm:
the candidate writes split scratch after nine shared stages, launches again,
then reads scratch for the radix-8 tail and writes interleaved output. cuFFT
uses one specialized kernel. This extra launch, global round trip, and the
shared-stage barriers explain the largest B64 gap; launch overhead dominates
the B1 gap. Eliminating wrapper copies cannot remove this algorithmic delta.

## Tuning and failed approaches

- Blind search: one independent radix-2/256-thread configuration.
- Production search: eight existing `(block_size, threads)` configurations on
  each of three rows (24 correctness-gated native-CUPTI cases), plus repeated
  official autotune runs. The official runs selected 256/256 on all rows; that
  common N=4096 default was retained.
- Conservative budget from recorded wall time: under 1.1 single-H200 GPU-hours
  for the complete round, including tests/profilers, with under 0.1 wall-clock
  hours spent compiling/autotuning configurations. No second GPU/container or
  parallel implementation agent was used.
- Replacing shared-stage trig with LUT lookup while leaving public split copies
  produced 0.1085/0.2556/0.2524 ms. It proved the arithmetic path but not a
  useful public-call optimization; direct interleaved input was required.
- A shared-memory radix-8 specialization regressed representative medians from
  roughly 0.0072/0.01184/0.01530 to 0.00914/0.01187/0.01712 ms and was removed.
- Combining PyTorch profiler and native CUPTI in one process disturbed CUPTI
  prepare/discovery. No fallback was accepted; profiling and final timing were
  split into separate processes. The failed attempt is recorded in the shared
  evidence errors.

Ideas adopted after incumbent inspection are explicitly: fused bit-reversal +
shared stages, radix-8/radix-4 global tails, the stage-indexed LUT/cache layout,
and final interleaved stores. New work in this round is the copy-free direct
interleaved input path and using the incumbent LUT in its shared stages. The
blind radix-2 runtime remains separate and is not relabeled as the production
implementation.

## Tests and blast radius

All GPU/Python commands below ran through `tileops-container.sh` and retain
JUnit XML under `worktree-artifacts/`:

| Command | Result | Artifact |
| --- | --- | --- |
| `pytest -q tests/ops/test_fft.py` | 18 passed | `required-fft.xml` |
| `pytest -q tests/kernels/test_fft_output_layout.py` | 4 passed | `required-layout.xml` |
| combined FFT/layout/workload post-cleanup run | 24 passed | `validation/post-cleanup-core.xml` |
| `pytest -q tests/test_validate_manifest.py` | 122 passed | `validation/post-cleanup-manifest.xml` |
| `pytest -q benchmarks/tests` | 22 passed | `validation/post-cleanup-benchmark-contract.xml` |
| `pytest -q -s benchmarks/ops/bench_fft.py` | 3 passed | `validation/post-cleanup-benchmark.xml` and `post-cleanup-profile_run.log` |
| authored HIR oracle | 4 passed | `authored-hir-oracle.xml` |
| final CUPTI comparison | 9 checks, zero failures | `final/final-evidence.xml` |
| separate PyTorch profiler | 9 checks, zero failures | `final/profiler-evidence.xml` |

Caller search found `FFTC2CKernel` consumers in `FFTC2COp`, the kernel layout
test, and exports; `FFTWorkload` consumers are the FFT op test and FFT benchmark.
All behavioral consumers were run. No shared signature or return shape changed:
the public kernel forward stacks split inputs into the new internal interleaved
entry, and custom kernel-map overrides retain their four-tensor call. This is
why unrelated 2,966-node smoke tests were not used as a substitute for the
computed blast radius.

`tests/ops/test_fft.py` grows from 9 to 18 nodes (+9, +100%). The increase is
justified by three exact manifest rows and distinct zero-stage, single-stage,
leading-dimension, strided-view, conjugate-view, and custom-kernel branches.
Ruff, `py_compile`, `git diff --check`, and GitHub pre-commit pass.

Exact final local commands (run from the TileOPs worktree) were:

```bash
$HOST_HOME/foreman/local/tileops-container.sh python -m pytest -q tests/ops/test_fft.py --junitxml=round5-artifacts/required-fft.xml
$HOST_HOME/foreman/local/tileops-container.sh python -m pytest -q tests/kernels/test_fft_output_layout.py --junitxml=round5-artifacts/required-layout.xml
$HOST_HOME/foreman/local/tileops-container.sh python -m pytest -q tests/ops/test_fft.py tests/kernels/test_fft_output_layout.py tests/test_workload_placement.py --junitxml=round5-artifacts/validation/post-cleanup-core.xml
$HOST_HOME/foreman/local/tileops-container.sh python -m pytest -q tests/test_validate_manifest.py --junitxml=round5-artifacts/validation/post-cleanup-manifest.xml
$HOST_HOME/foreman/local/tileops-container.sh python -m pytest -q benchmarks/tests --junitxml=round5-artifacts/validation/post-cleanup-benchmark-contract.xml
$HOST_HOME/foreman/local/tileops-container.sh python -m pytest -q -s benchmarks/ops/bench_fft.py --junitxml=round5-artifacts/validation/post-cleanup-benchmark.xml
$HOST_HOME/foreman/local/tileops-container.sh env PYTHONPATH=$CONTAINER_WORKSPACE/tileops python scripts/fft_final_evidence.py --base-source round5-artifacts/base_fft.py --artifact-dir round5-artifacts/final --trials 3
$HOST_HOME/foreman/local/tileops-container.sh env PYTHONPATH=$CONTAINER_WORKSPACE/tileops python scripts/fft_profiler_evidence.py --base-source round5-artifacts/base_fft.py --artifact-dir round5-artifacts/final
```

The successful blind command and exact host TileFoundry CLI invocations are
recorded verbatim in `first-candidate.md`; their XML/JSON outputs remain in the
shared round root. Evidence scripts were copied to `evidence-sources/` before
the untracked worktree copies were removed.

## TileFoundry gaps

Detailed structured records and reproducers are in `tilefoundry-gaps.md`:

1. `TF-R5-FFT-01`, `semantic-blocker`, new: current HIR rejects direct
   `complex64` and `complex128`. The f32 pair workaround describes c64
   arithmetic but not the exact public tensor type.
2. `TF-R5-FFT-02`, `semantic-blocker`, new and distinct: current HIR rejects
   `f64`, so a paired-real exact complex128 graph is also unavailable. The
   handwritten TileLang workaround is correct and measures 0.015296 ms, but
   cannot supply exact authored-HIR provenance.

Tuple concat, static FFT-stage layout, evaluator, check/analyze/schedule, and
canonical round-trip succeeded. No dynamic stage indexing was required, so no
rounds 3/4 `RangeSlice` duplicate is claimed. cuFFT performance and NCU host
permissions are explicitly non-gaps.

## CI and remaining risks

- PR #1892 is open and GitHub reports `MERGEABLE`. Every executed check is
  green: GPU smoke, security policy, compile-contract gate,
  benchmark-contract tests, pre-commit, packaging, gitleaks, actionlint, title,
  manifest stats, and supporting gates. `validate-manifest` was correctly
  skipped because no manifest file changed; the local 122-test validator run
  passed. The PR was not merged.
- GPU Smoke run `31543589115` passed in 4m49s. Its broad smoke JUnit records
  2,958 tests (2,944 passed, 14 skipped, zero failures); its diff-scoped full
  JUnit records all 18 FFT nodes passing. Raw logs, reports, and both XML files
  are under `ci-artifacts/gpu_smoke.log/`. `ci-artifacts/final-checks.log`,
  `pr-final.json`, and `preflight-run.log` retain final CI/PR evidence.
- SHA256: broad smoke XML
  `bf1e628f48f16ffa633443e54d634cd711958e44eb274a48e403eaca305fb9ab`;
  diff-scoped XML
  `59874c3a48631e0f47520bab7050f5cf52596012d84d218ab26d89bd39da107f`.
- The fast path intentionally targets the exact contiguous manifest inputs.
  Strided inputs materialize one interleaved contiguous tensor; conjugate views
  and custom kernels use the preserved split path. Correctness is covered, but
  those fallback layouts were not performance objectives.
- The B1 trial medians exhibit launch-sensitive outliers. Raw samples are
  retained, and both the base-improvement and cuFFT-deficit conclusions remain
  separated even under the favorable endpoint comparisons described above.
- Exact register allocation and hardware counters remain unknown because host
  NCU permissions are unavailable. Launch traces, generated source, memory
  accounting, and controlled ablations support the stated bottleneck without
  inventing counter data.

## CUDA 13.2 official-runner retest

Retest date: 2026-08-12 (Asia/Shanghai). The unchanged PR commit
`bf8d0599325209319c365a3488338ec6905691d6` was retested in the new official
schema-3 persistent container after explicit GPU sequencing approval.

- Image: `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`,
  digest `sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`.
- PyTorch 2.13.0+cu132; CUDA 13.2; TileLang
  `0.1.11+cu132.gitafcebed1`; `cupti-python` 13.2.0;
  `cuda-bindings` 13.3.1.
- GPU: NVIDIA H200, capability 9.0, driver 595.71.05. No other compute process
  was present when the retest began. No dependency was installed or replaced.

### Correctness and compatibility

The first combined cold-container run aborted once inside TileLang's CUDA pass
pipeline while compiling the eighth specialization. It produced a native
`Fatal Python error: Aborted`, not a numerical assertion or CUDA runtime error.
Six isolated public-call reproductions then compiled and passed: N=1024/B1 c64,
N=1024/B8 c64, N=128/B4 c128, and all three N=4096 manifest rows. The complete
18-node FFT suite subsequently passed, followed by a combined 24-node
FFT/layout/workload run. The abort did not recur during 24 autotune compiles,
base/candidate compilation, profiling, or final evidence collection. This is
retained as a one-time cold-compile environment observation; there is no
reproducible kernel compatibility defect to patch.

| CUDA 13.2 command | Result | Artifact under `cu132-retest/` |
| --- | --- | --- |
| `pytest -q tests/ops/test_fft.py` | 18 passed | `validation/fft-rerun.{log,xml}` |
| combined FFT/layout/workload run | 24 passed | `validation/core-rerun.{log,xml}` |
| `pytest -q tests/test_validate_manifest.py` | 122 passed | `validation/manifest.{log,xml}` |
| `pytest -q benchmarks/tests` | 22 passed | `validation/benchmark-contract.{log,xml}` |
| `pytest -q -s benchmarks/ops/bench_fft.py` | 3 passed; 256/256 selected on all rows | `validation/benchmark.{log,xml}` and `profile_run.log` |
| native-CUPTI three-way comparison | 9 checks, zero failures | `final/final-evidence.{json,xml,log}` |
| PyTorch profiler | 9 checks, zero failures | `final/profiler-evidence.{json,xml,log}` |

`core-rerun.xml` SHA256 is
`22c25bb569187fb4fd4ba52bfbced52ec74abc261273b649e6326c4b5c634627`.

### CUDA 13.2 performance

The original same-process protocol was repeated unchanged: exact same input
per row, fixed-base source (SHA256
`8c64d6f3f0e5f0e8ba08e7459e0951c325daf0a6d64f33d27658eaf0b961c533`),
warmed candidate/base/cuFFT, cuFFT plan cache 0 to 3, three position rotations,
200 native-CUPTI samples per trial, and no event fallback.

| Row | Candidate trial medians ms | Base trial medians ms | cuFFT trial medians ms |
| --- | --- | --- | --- |
| 4K / B1 / c64 | 0.0068800, 0.0067515, 0.0068150 | 0.1051040, 0.1040640, 0.1035680 | 0.0046080, 0.0045920, 0.0046080 |
| 4K / B64 / c64 | 0.0108320, 0.0098240, 0.0097920 | 0.1250400, 0.1157920, 0.1149440 | 0.0048320, 0.0048640, 0.0048325 |
| 4K / B64 / c128 | 0.0131200, 0.0131200, 0.0130880 | 0.1325925, 0.1343365, 0.1287200 | 0.0073920, 0.0073920, 0.0073920 |

Median-of-three and same-environment ratios:

| Row | Candidate ms | Base ms | cuFFT ms | Base / candidate | Candidate / cuFFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K / B1 / c64 | 0.0068150 | 0.1040640 | 0.0046080 | 15.2698x | 1.4789x slower |
| 4K / B64 / c64 | 0.0098240 | 0.1157920 | 0.0048325 | 11.7866x | 2.0329x slower |
| 4K / B64 / c128 | 0.0131200 | 0.1325925 | 0.0073920 | 10.1061x | 1.7749x slower |
| geometric mean | 0.0095770 | 0.1169049 | 0.0054804 | 12.2068x | 1.7475x slower |

Old CUDA 12.9 versus new CUDA 13.2 median-of-three comparison:

| Row | Candidate old -> new | Base old -> new | cuFFT old -> new |
| --- | ---: | ---: | ---: |
| 4K / B1 / c64 | 0.0072165 -> 0.0068150 (-5.56%) | 0.1155360 -> 0.1040640 (-9.93%) | 0.0051680 -> 0.0046080 (-10.84%) |
| 4K / B64 / c64 | 0.0120960 -> 0.0098240 (-18.78%) | 0.1084005 -> 0.1157920 (+6.82%) | 0.0056320 -> 0.0048325 (-14.20%) |
| 4K / B64 / c128 | 0.0152960 -> 0.0131200 (-14.23%) | 0.1288805 -> 0.1325925 (+2.88%) | 0.0084800 -> 0.0073920 (-12.83%) |
| geometric mean | 0.0110116 -> 0.0095770 (-13.03%) | 0.1173038 -> 0.1169049 (-0.34%) | 0.0062728 -> 0.0054804 (-12.63%) |

This is a stack-to-stack observation, not a cross-environment speedup claim.
Within CUDA 13.2, the candidate remains materially faster than the fixed base
on every row but slower than cuFFT on every row. The classification therefore
remains **improvement without SOTA**.

Correctness errors are bit-for-bit the same values recorded under CUDA 12.9:
candidate max/mean complex absolute errors are
`4.264961e-5 / 1.090792e-5`, `5.554283e-5 / 1.082392e-5`, and
`1.994589e-13 / 3.616139e-14`; every `assert_close` passes. Launch and memory
structure also remains unchanged: candidate 2 kernels, base 2 copy + 2 FFT
kernels, cuFFT 1 kernel; candidate temporary storage is half the base on every
row. PyTorch profiler event counts remain 12/23/13 for
candidate/base/cuFFT.

The CUDA 13.2 final JSON SHA256 is
`3ea1d0843cc46d3f60337522344702bbd573b0b5f3459ec4cf5458e37b827f48`;
the zero-failure JUnit SHA256 is
`43a8c6b29f22a2ee92aedba34bf03ec2da874ba1b0c1a2babf68a06feae65a9a`.
Raw samples, generated CUDA, CUPTI launch traces, profiler traces, old/new JSON,
logs, and JUnit files are all retained under `cu132-retest/`.

No TileOPs source or test change was necessary, so PR #1892 remains at its
original commit and the PR body/performance claim remains conservative. GitHub
automatically restarted GPU Smoke run `31543589115` as attempt 2 during this
retest. All non-GPU jobs passed; its `gpu-smoke` job `93959481829` remained
queued with no runner assigned (`runner_name` empty) when reporting completed.
It was neither cancelled nor restarted. The unchanged commit's attempt-1 GPU
Smoke had already passed 2,958 broad nodes and all 18 diff-scoped FFT nodes;
the independent official CUDA 13.2 container evidence above supplies the new
stack result while the shared self-hosted queue waits.

## Public PR contract migration

On 2026-08-12 the public PR metadata was regenerated against TileOPs Foundry
Loop commit `a1ef642fcb325cbd63fdb2e81045837ad3560ae8`. The renderer consumed the
structured PR record and extracted the unique top-level `@module` class from
the authored HIR; its checker passed before publication.

- Public title: `[Perf][foundry][FFT] Remove split-input copies from C2C FFT`.
- Public body has exactly `Summary`, `TileFoundry Description`, `Performance`,
  and `Result And Limitations`, in that order.
- The performance table uses the CUDA 13.2 median-of-three results above for
  every primary row and all three runnable comparators: candidate, fixed-base
  incumbent, and PyTorch cuFFT.
- Classification remains **improvement without SOTA**. Per-row incumbent
  speedups, cuFFT deficits, and trial-median spans are public; internal
  correctness, reproduction, artifact, filename, and local-path material is
  absent.
- GitHub read-back matches the generated body modulo the API-added final
  newline. A separate leak scan and exact extracted-class comparison passed.

The metadata-only update left HEAD at
`bf8d0599325209319c365a3488338ec6905691d6`. The queued GPU Smoke attempt 2
subsequently completed successfully in 6m18s, and the title-edit label job also
passed. At final verification every non-skipped check was successful, no check
was pending, GitHub reported `mergeable=MERGEABLE`, and the only remaining
`mergeStateStatus=BLOCKED` condition was the requested but not-yet-submitted
`tileops-review` human review. The PR remains open and was not merged.

## Rebase onto main 994f43d

On 2026-08-12 the round branch was rebased from base
`5c4d54c44dc60a3bee5bf2b409cf224b7f16c820` onto requested main
`994f43d374cbc1d77a08e207ab8831402b87a376`. The worktree was clean before
starting: no staged, unstaged, or untracked files existed. The old HEAD
`bf8d0599325209319c365a3488338ec6905691d6` was preserved as local branch
`backup/perf-tileops-r5-fft-c2c-pre-rebase-20260812`, and the public PR JSON
was captured before mutation.

The single round commit replayed without conflicts as
`9f4cc1f2c489ef139d621662494b98789089429a`. `git range-diff` reports the old
and new commits as patch-equivalent, and the four round-owned implementation,
op, test, and workload files are byte-identical across the rebase. Main did,
however, change the FFT benchmark to `ManifestBenchmark.compare`, split the
timing module, replace sequence attribution with per-call CUPTI windows, add
forward/reverse drift balancing, and pin the CUDA 13.2 runner. This changed
benchmark behavior, so performance was refreshed rather than carried forward.

### Post-rebase validation

- Manifest and complete benchmark contract suites: 140 tests, 0 failures, 0
  errors, run twice. JUnit SHA256:
  `b2f1de008dd124368c05b824b7f7ff3facd6b540bdbcaa58701b272abe82aa0a`.
- FFT op, output-layout, and workload-placement blast radius: 24 passed in
  9.46s. JUnit SHA256:
  `ba9d2db3374ee822f3d946004445a2758a054296a60e534351ee721f628a778e`.
- Official FFT benchmark: 3 passed in 5.07s; every row selected
  `block_size=256, threads=256`. JUnit SHA256:
  `4a020e7042213b08141a0edb65aa6485ca309341c2d011351cec6bde055b6a6a`.
- Test-node delta remains 9 to 18 (+9, +100%). Ruff, `py_compile`,
  `git diff --check`, Foundry Loop render/check, and patch-equivalence checks
  pass.

All TileOPs commands ran through the persistent official CUDA 13.2 container;
GPU compilation and measurement were serialized under the GPU 3 lock. No
dependency was installed or replaced.

### Post-rebase performance

The fixed incumbent remained the exact pre-round source at `5c4d54c`, with
only its custom-op symbol renamed so it could coexist with the candidate. The
new harness used `ManifestBenchmark.compare`, native-CUPTI call-window
attribution, L2 reset, forward/reverse drift balancing within each comparison,
three rotated three-way insertion orders, and 200 samples per implementation
per trial. Compilation, tuning, LUT creation, and cuFFT plan creation remained
outside timing. Candidate, incumbent, and cuFFT correctness passed on every
row.

| Row | Candidate trial medians ms | Incumbent trial medians ms | cuFFT trial medians ms |
| --- | --- | --- | --- |
| 4K / B1 / c64 | 0.0066400, 0.0065600, 0.0066405 | 0.1223515, 0.1181600, 0.1046400 | 0.0046080, 0.0045770, 0.0046080 |
| 4K / B64 / c64 | 0.0097600, 0.0097920, 0.0097920 | 0.1077760, 0.1042245, 0.1077920 | 0.0048640, 0.0048960, 0.0048960 |
| 4K / B64 / c128 | 0.0128960, 0.0129280, 0.0129280 | 0.1246720, 0.1125760, 0.1208480 | 0.0073920, 0.0073920, 0.0073920 |

Median-of-three results:

| Row | Candidate ms | Incumbent ms | cuFFT ms | Incumbent / candidate | Candidate / cuFFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K / B1 / c64 | 0.006640 | 0.118160 | 0.004608 | 17.7952x | 1.4410x slower |
| 4K / B64 / c64 | 0.009792 | 0.107776 | 0.004896 | 11.0065x | 2.0000x slower |
| 4K / B64 / c128 | 0.012928 | 0.120848 | 0.007392 | 9.3478x | 1.7489x slower |
| geometric mean | 0.0094375 | 0.1154544 | 0.0055043 | 12.2336x | 1.7146x slower |

Trial-median spans for candidate/incumbent/cuFFT are 1.21%/14.99%/0.67%,
0.33%/3.31%/0.65%, and 0.25%/10.01%/0.00%. Even the favorable endpoints keep
the candidate ahead of the incumbent and behind cuFFT on every row. The honest
classification therefore remains **improvement without SOTA**.

The comparison JSON SHA256 is
`8ad5da36eebcbbe74f04377e39be4940ae0d941eab7ffa38899c5fe62c4d69f7`;
its zero-failure 36-case JUnit SHA256 is
`3bcae75fe424e6f911f0129306c57fcb8ff546b3508f2779a0af048b5d0c4f9c`.
Both, the benchmark report, the post-rebase harness, and all validation JUnit
files are retained under the round's internal `post-rebase` evidence tree.
The structured PR data was updated with these measured values and regenerated;
its title and four-section public contract remain unchanged.
