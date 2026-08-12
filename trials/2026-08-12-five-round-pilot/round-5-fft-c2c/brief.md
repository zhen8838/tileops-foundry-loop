# Round 5 brief: FFTC2COp

## Objective

Produce a correct TileLang implementation derived through a TileFoundry authored-HIR workflow
for `FFTC2COp`, then optimize all three 4096-point manifest workloads on NVIDIA H200. The
strongest runnable same-contract external baseline is CUDA PyTorch `torch.fft.fft`, backed by
cuFFT. Take an independently useful result through a mergeable TileOPs PR; if no honest
performance PR is justified, finish with reproducible evidence and concrete blockers.

This is round 5 of the fixed five-round sequence. Do not use or substitute a GQA operator. You
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

  The first call starts the container; later calls use `docker exec` in that same container on
  the same GPU. `shell` enters it interactively. Do not create another ad hoc container.
- Inside the container, `$CONTAINER_WORKSPACE/tileops` is writable, `$CONTAINER_WORKSPACE/tilefoundry` is read-only,
  and `$CI_CACHE` persists. The common TileOPs Git directory is mounted read-only and
  `GIT_OPTIONAL_LOCKS=0` is set.
- Use TileFoundry on the host only through
  `$HOST_HOME/TileFoundry/.venv/bin/<tool>`. Do not rebuild that environment.
- Do not install, upgrade, or replace packages. The image has CUDA 12.9, Python 3.12, PyTorch
  2.10.0+cu129, TileLang 0.1.11+cu129.gitafcebed1, and `cupti-python==12.8.0`.
- Native CUPTI timing is required for final latency. NCU performance counters are unavailable on
  this host with `ERR_NVGPUCTRPERM`; use CUPTI, PyTorch profiler, generated-source inspection,
  and controlled ablations.
- Shared artifacts belong under
  `$TRIAL_SOURCE/round-5-fft-c2c/`.
- `preflight.log` in that directory records the structural manifest, exact output contracts,
  cuFFT plan-cache creation, baseline profiler kernel names, correctness, and initial event timing
  for all three primary rows.

## Authoritative sources and blind boundary

Before producing the first correct candidate, read only these contract surfaces:

- Load `FFTC2COp` structurally through `tileops.manifest`; its entry is in
  `src/tileops/manifest/sequence_modeling.yaml`.
- `workloads/fft.py`: input generation and the `torch.fft.fft` reference.
- `src/tileops/ops/fft.py`: public constructor/forward signature, validation, shape flattening,
  twiddle-cache input contract, and dispatch surface only. Do not follow the kernel import.
- `tests/ops/test_fft.py`: public correctness cases and tolerances.
- `benchmarks/ops/bench_fft.py` and `benchmarks/benchmark_base.py`.
- CUDA PyTorch `torch.fft.fft`, cuFFT plan-cache state, and profiler-visible baseline behavior.
- TileFoundry specs, tutorial, CLI, authored-HIR examples, and unrelated TileLang kernels needed
  only for repository conventions.

Until `first-candidate.md` and every artifact it cites are written, do not open, print, search
within, import-source-inspect, disassemble, or otherwise read `src/tileops/kernels/fft.py`,
generated/cached copies of that kernel, or `tests/kernels/test_fft_output_layout.py` (which exposes
incumbent configuration/radix details). Do not use broad searches that cross those paths. Record
all blind-phase source reads and the exact timestamp when blindness ends.

Before ending the blind phase, the new implementation must:

1. originate from an authored TileFoundry HIR description and a traceable TileLang runtime twin;
2. pass a small multi-stage complex64 case against an independent direct DFT oracle;
3. pass the complete unbatched `N=4096` complex64 manifest row against `torch.fft.fft` at the
   existing tolerance; and
4. record at least three native-CUPTI trials for that 4096-point row.

Save the authored HIR, current TileFoundry check/analyze/schedule results, minimal dtype failures,
runtime source, generated CUDA, raw samples, read audit, errors, and limitations in
`first-candidate.md`. Make it immutable or record a checksum before reading incumbent source.
Afterward, incumbent inspection is permitted for diagnosis and optimization; identify every idea
adopted from it. Never relabel an incumbent-derived path as TileFoundry-derived.

## Public operator contract

`FFTC2COp.forward(input)` computes a one-dimensional complex-to-complex FFT along the final axis.

| Value | Shape | Dtype / rule |
| --- | --- | --- |
| `input` | `[...,N]` | complex64 or complex128, CUDA |
| `output` | same as input | same dtype as input |

- `input.ndim >= 1`, `N > 0`, and `N` is a power of two.
- Any leading dimensions are batch dimensions; their product is the flat runtime batch.
- The transform is forward, unnormalized (`norm="backward"` default), with the natural-frequency
  output order used by `torch.fft.fft`.
- No conjugation, inverse sign, scaling, padding, truncation, or real-FFT packing is allowed.
- Preserve the public validation errors, device behavior, arbitrary leading dimensions, kernel
  and twiddle caching, and `tune`/`kernel_map` behavior.

The primary manifest inputs are contiguous complex tensors. Any real/imag extraction, packing,
contiguous conversion, interleaved-layout conversion, and output materialization required by a
candidate is part of the public call and must remain inside timing. Shape/dtype/device-only plans
or twiddles may be cached after warmup; runtime input values may not be precomputed or cached.

A production specialization may target exactly the manifest rows while other supported cases
retain their current behavior. Add correctness coverage for all primary rows plus boundary cases
such as `N=1`/`N=2`, nontrivial leading dimensions, and a non-contiguous supported view or prove
the precise current public restriction. Do not weaken existing cases.

## Mathematical definition

For each independent leading-index batch and output frequency `k`, with `0 <= k < N`:

```text
theta(t,k) = -2*pi*t*k/N
W_real(t,k) = cos(theta(t,k))
W_imag(t,k) = sin(theta(t,k))

Y_real[k] = sum_t (X_real[t] * W_real(t,k) - X_imag[t] * W_imag(t,k))
Y_imag[k] = sum_t (X_real[t] * W_imag(t,k) + X_imag[t] * W_real(t,k))
Y[k] = complex(Y_real[k], Y_imag[k])
```

This direct DFT is the semantic oracle. A radix-2/4/8 FFT, Stockham transform, or another
Cooley-Tukey factorization is equivalent only if it produces the same natural-order output and
normalization. Twiddle tables computed only from `(N,dtype,device)` are allowed and must use
enough precision: float32 components for complex64 and float64 for complex128.

Use the existing tolerances without loosening them: complex64 `atol=rtol=1e-4`; complex128
`atol=rtol=1e-8`. Report maximum and mean complex absolute error as well as the actual
`torch.testing.assert_close` result. Test exact shapes/dtypes, not just one component.

## TileFoundry authored-HIR description

First attempt the exact public tensor types (`complex64` and `complex128`) in current authored
HIR and retain the smallest parser/type error. The current published dtype spec appears not to
list complex or f64; this observation is not enough to declare a gap. Demonstrate it against the
checkout.

Then author a real-pair HIR for the representable complex64 semantics. At minimum include a
typed complex butterfly and a small direct DFT or FFT stage. Precomputed twiddles are explicit
inputs so missing trigonometric authoring does not change the transform contract:

```python
@func
def complex_butterfly(
    even_r: Tensor[(B,M), "f32"], even_i: Tensor[(B,M), "f32"],
    odd_r: Tensor[(B,M), "f32"], odd_i: Tensor[(B,M), "f32"],
    tw_r: Tensor[(M,), "f32"], tw_i: Tensor[(M,), "f32"],
) -> tuple[Tensor[(B,2*M), "f32"], Tensor[(B,2*M), "f32"]]:
    prod_r = odd_r * tw_r - odd_i * tw_i
    prod_i = odd_r * tw_i + odd_i * tw_r
    # Place even +/- product in the stage's declared frequency layout.
    return pair_concat(even_r + prod_r, even_r - prod_r), pair_concat(even_i + prod_i, even_i - prod_i)

@func
def dft_pair_f32(
    x_r: Tensor[(B,N), "f32"], x_i: Tensor[(B,N), "f32"],
    w_r: Tensor[(N,N), "f32"], w_i: Tensor[(N,N), "f32"],
) -> tuple[Tensor[(B,N), "f32"], Tensor[(B,N), "f32"]]:
    y_r = matmul(x_r, w_r) - matmul(x_i, w_i)
    y_i = matmul(x_r, w_i) + matmul(x_i, w_r)
    return y_r, y_i
```

Adapt syntax, orientation, and layout to the current DSL, and validate against a small known DFT.
Exercise `tilefoundry check`, `analyze`, `schedule`, parser/evaluator, and source-to-source
surfaces as applicable. If tuple concatenation or FFT-stage index permutation fails, write the
smallest concrete reproducer and compare it with rounds 3/4's loop `RangeSlice` gap before
calling it new.

For complex128, an f32 pair is not an exact HIR substitute. Attempt an f64 real-pair description
and retain the exact failure. A semantics-preserving handwritten TileLang runtime twin may bridge
an HIR limitation only if the limitation and workaround are explicit; never claim the f32 HIR
describes complex128 precision.

The optimized TileLang runtime may use shared-memory or register butterflies, warp shuffles,
mixed radices, batched CTAs, direct interleaved complex storage, or a multi-stage pipeline.
Every stage must remain traceable to the DFT graph. It must not call `torch.fft`, cuFFT, or the
incumbent kernel, and must not precompute any input-dependent transform.

## Manifest distribution

Benchmark every primary row:

| Label | Logical shape | Flat batch | N | dtype |
| --- | --- | ---: | ---: | --- |
| `fft-4k-c64-unbatched` | `[4096]` | 1 | 4096 | complex64 |
| `fft-4k-c64-b64` | `[64,4096]` | 64 | 4096 | complex64 |
| `fft-4k-c128-b64` | `[64,4096]` | 64 | 4096 | complex128 |

Tune across all three rows. Do not exclude complex128, the unbatched launch-sensitive case, or a
slower row from the SOTA criterion.

## Equivalent cuFFT baseline

For each row use the exact same CUDA tensor and call:

```python
torch.fft.fft(input, dim=-1)
```

Use the default forward normalization. Warm the exact shape/dtype/device plan before steady-state
timing and record `torch.backends.cuda.cufft_plan_cache` state. Do not include plan creation in
cuFFT steady-state latency while excluding TileLang compilation/tuning from the candidate.
Conversely, do not move any per-call input/output conversion out of candidate timing.

The preflight proves all three baseline calls execute, preserve shape/dtype, and create distinct
cuFFT plan-cache entries. Profiler evidence for complex64 exposes the CUDA `vector_fft<4096...>`
kernel. Initial event medians, which are observations rather than final evidence, are:

| Row | incumbent (`tune=False`) | `torch.fft.fft` |
| --- | ---: | ---: |
| 4K / B1 / complex64 | 0.3681 ms | 0.0475 ms |
| 4K / B64 / complex64 | 0.4444 ms | 0.0706 ms |
| 4K / B64 / complex128 | 0.1213 ms | 0.0207 ms |

Re-run the final candidate, base-commit incumbent, and cuFFT with native CUPTI and a common
interleaved harness. A fallback or CPU FFT never counts as performance evidence.

## Performance and SOTA contract

Report final candidate, base-commit incumbent, and cuFFT in the same persistent container,
process, GPU, exact inputs, precision contract, warmup policy, and native-CUPTI harness. Use
rotated/interleaved repeated trials to quantify drift. Exclude compilation, tuning, LUT creation,
and cuFFT plan creation from steady state; include every per-call transform and synchronize
correctly. Retain all raw samples and `profile_run.log`.

SOTA requires the candidate to be no slower than cuFFT within measured noise on every one of the
three rows and to have a strictly lower geometric-mean latency. Also report per-row and
geometric-mean speedups against the base incumbent. A tie, a statistically indistinguishable
result, or any primary-row regression is not SOTA.

Do not change normalization, use different inputs, reduce batch/N, lower precision, time only
the butterfly while excluding public-call copies, reuse outputs, precompute runtime values,
dispatch to cuFFT/incumbent, or count less work. Record configuration count, compile-hours,
GPU-hours, and selection method. If SOTA is not reached, retain a candidate only when it is
independently useful and reviewable.

## Required execution and evidence

1. Complete and seal the two-phase blind evidence before incumbent inspection.
2. Run at least:

   ```bash
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/ops/test_fft.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q tests/test_validate_manifest.py
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q benchmarks/tests
   $HOST_HOME/foreman/local/tileops-container.sh \
     python -m pytest -q -s benchmarks/ops/bench_fft.py
   ```

   After the blind seal, include `tests/kernels/test_fft_output_layout.py` and broader suites in
   proportion to shared-code changes.
3. Correctness-test all three manifest rows against cuFFT and small direct DFT cases against an
   independent oracle. Cover output order, normalization, both precisions, batch flattening, and
   boundary/layout behavior.
4. Profile all three rows. Attribute public-wrapper copies/views, launch count, temporary memory,
   FFT kernel time, shared/register resources from generated source, and a concrete bottleneck.
5. Prove measured rows execute the new candidate without cuFFT/incumbent dispatch or
   input-dependent precomputation.
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

Treat direct complex dtype support and f64 real-pair precision as related but distinct facts if
their reproducers/affected rows differ. Compare any dynamic stage-index issue with the round 3/4
`RangeSlice` gap. Do not call cuFFT's specialized performance or every missing FFT convenience
operator a TileFoundry bug.

## Final round report

Write `$TRIAL_SOURCE/round-5-fft-c2c/report.md` containing:

- branch, commits, PR URL/status, container/GPU, and exact versions;
- blind-phase read audit/timestamp, authored HIR validation, checksum, and first candidate;
- exact correctness/test commands and artifact paths;
- raw latency/error/noise tables for all three rows, candidate, base incumbent, and cuFFT, with
  geometric means and speedups;
- explicit classification: `measured SOTA`, `improvement without SOTA`, or `no improvement`;
- profiler/copy/temporary-memory/launch evidence, tuning budget, failed approaches, incumbent
  ideas adopted, and remaining risks;
- structured TileFoundry gaps with minimal reproducers and measured workaround costs.

Do not finish before a useful PR is mergeable, or the report proves with reproducible evidence
why no honest PR can be opened.
