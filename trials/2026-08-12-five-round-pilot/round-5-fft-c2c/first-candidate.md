# Round 5 blind first candidate

The blind phase ended at `2026-08-12T05:57:26.225141650+08:00` (Asia/Shanghai).
This file and all cited artifacts were written before reading, searching within,
import-source-inspecting, disassembling, or otherwise opening
`src/tileops/kernels/fft.py`, any generated/cache copy of it, or
`tests/kernels/test_fft_output_layout.py`.

## Draft corrections established before implementation

- `load_manifest()` returns the merged op dictionary directly; the authoritative
  structural read is `load_manifest()["FFTC2COp"]` and confirms exactly three
  4096-point rows.
- The existing op materializes `input.real.contiguous()` and
  `input.imag.contiguous()` before kernel launch. Final public-call timing must
  include both copies; kernel-only timing cannot establish this round's result.
- The repository benchmark framework already performs native-CUPTI discovery,
  complete-sequence attribution, L2 reset, shifted input addresses, and fail-closed
  behavior. Final interleaved comparisons will extend that framework instead of
  introducing a CUDA-event timing contract.
- Existing unit cases stop at N=1024 and omit N=1/N=2, all three manifest rows,
  and a non-contiguous view. Contract evidence therefore needs both curated
  branch/boundary tests and a separate exact-manifest correctness artifact.
- Current TileFoundry authored HIR uses `tf.concat(a, b, axis=...)`,
  `tf.matmul`, tuple returns, `@module`, and explicit target/topology declarations;
  the brief pseudocode was adapted to these actual surfaces without changing the
  DFT orientation or arithmetic.

## Authored HIR and demonstrated dtype boundaries

`authored_hir.py:FFTPairF32` is an authored f32 real/imaginary-pair description.
It contains a typed complex butterfly with explicit twiddle inputs and a direct
DFT with `W[t,k]` orientation, returning natural-frequency real/imaginary pairs.

- `authored-hir-check.json` passes `nan_inf` for both f32 `[2,4]` outputs.
- `authored-hir-oracle.json` and `authored-hir-oracle.xml` compare the evaluator
  to literal butterfly and direct-DFT equations. All four checks pass; the largest
  error is `1.1920928955078125e-07`.
- `authored-hir-analyze.json` completes compute-cost, memory, roofline, and
  timeline analysis. The fixed DFT graph reports 272 f32 flops, 512 bytes read,
  and 192 bytes written.
- `authored-hir-schedule.json` records a six-operation H200 CTA plan with
  `FEASIBLE_NOT_PROVEN` and objective 6 ns. This is schedule advice, not measured
  runtime.
- `authored_hir_roundtrip.py` is canonical source emitted from the authored
  Module and successfully re-imported, exercising the source-to-source surface.
- `tilefoundry-cli.xml` contains six CLI cases with zero unexpected failures.

The exact-type attempts are separate minimal reproducers and failures:

| Reproducer | Actual result | Interpretation |
| --- | --- | --- |
| `complex64_dtype_repro.py` | unknown `complex64`; valid dtype list emitted | exact public c64 tensors are not representable |
| `complex128_dtype_repro.py` | unknown `complex128`; valid dtype list emitted | exact public c128 tensors are not representable |
| `f64_pair_repro.py` | unknown `f64`; valid dtype list emitted | an exact real-pair c128 graph is independently blocked |

The exact stderr is retained in `complex64-dtype.stderr`,
`complex128-dtype.stderr`, and `f64-pair.stderr`. The f32 HIR is not claimed to
describe complex128 precision.

Authored HIR SHA256 is
`14b7d96cd162a0e5ae8eeddd715fff2d222feeebd62f9e6556554519ad0b9056`.

## Graph-traceable TileLang runtime twin

The source is `evidence-sources/fft_blind_candidate.py`; generated CUDA is
`evidence-sources/fft_blind_candidate.generated.cu`.

The runtime is an iterative radix-2 decimation-in-time FFT. It loads a cached
shape-only bit-reversal permutation into two shared-memory real/imaginary arrays,
then executes twelve stages of disjoint complex butterflies using the same
stage-indexed twiddle layout stated by the authored HIR. The output is stored as
an interleaved real pair and exposed through metadata-only `view_as_complex`.
Input real/imaginary materialization remains inside `BlindFFT.__call__` and hence
inside timing. It does not call `torch.fft`, cuFFT, or any incumbent path, and it
caches no input-dependent value.

The generated CUDA contains one `fft_radix2_kernel`, launch bounds of 256 threads,
32 KiB dynamic shared memory (16 KiB for each component), a 12-stage butterfly
loop, and direct interleaved output stores. The source SHA256 values are:

```text
f1b9d5481b682734f8e2a87aec86727e534e3aa6e8dd9c059fad6db45324367c  fft_blind_candidate.py
d62085eb73af2399ef973fb44133b92d4a1c25b314839dbae0767fd2af7e2001  fft_blind_candidate.generated.cu
```

Only one production configuration was evaluated: radix 2, 256 threads, one CTA
per flat batch. Compilation plus evidence collection was under one minute and
well below 0.02 H200 GPU-hours. `blind-errors.log` retains the validation-device
mistake, missing `PYTHONPATH` launch, and one small-N compiler warning.

## Correctness and native-CUPTI timing

The successful command was:

```bash
$HOST_HOME/foreman/local/tileops-container.sh env \
  PYTHONPATH=$CONTAINER_WORKSPACE/tileops python scripts/fft_blind_candidate.py \
  --bench-trials 3 \
  --json-out $CONTAINER_WORKSPACE/tileops/round5-artifacts/blind-first-candidate.json \
  --junit-out $CONTAINER_WORKSPACE/tileops/round5-artifacts/blind-first-candidate.xml \
  --source-out $CONTAINER_WORKSPACE/tileops/round5-artifacts/fft_blind_candidate.generated.cu
```

Structured results and every raw sample are in
`artifacts/blind-first-candidate.json`; the two-case, zero-failure JUnit is
`artifacts/blind-first-candidate.xml`.

| Fixture | Oracle | max complex abs | mean complex abs | assert_close |
| --- | --- | ---: | ---: | --- |
| N=8, B=2, c64 (3 stages) | literal direct DFT | 5.663382e-6 | 2.021699e-6 | pass at 1e-4/1e-4 |
| N=4096, B=1, c64 | `torch.fft.fft` | 4.533716e-5 | 1.112630e-5 | pass at 1e-4/1e-4 |

| CUPTI trial | samples | median ms | p10 ms | p90 ms | timing |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 200 | 0.0688645 | 0.058369 | 0.076961 | cupti |
| 1 | 200 | 0.0727840 | 0.058336 | 0.076512 | cupti |
| 2 | 200 | 0.0700160 | 0.058368 | 0.076064 | cupti |

Median-of-trial-medians is 0.070016 ms. The alternating lower/higher samples
show input-address sensitivity in the public two-copy path; this first candidate
is correctness evidence, not a performance claim.

## Limitations

- One CTA per transform leaves the unbatched row launch/synchronization bound;
  generated CUDA inserts synchronization around the inner stage work.
- The f32 pair HIR cannot state c128 precision, and the first runtime evidence
  intentionally covers only the brief-mandated c64/B1 production row. All three
  rows, including c128/B64, remain for post-blind optimization and validation.
- No incumbent-derived configuration or radix has been used. Any idea adopted
  after the timestamp above will be identified separately.

## Blind source-read audit

The source phase began at approximately `2026-08-12T05:42+08:00`; all times are
Asia/Shanghai. Reads before the sealed timestamp, in order (parallel batches are
grouped), were:

```text
brief.md; CLAUDE.md
.claude/domain-rules/{ops-design,testing-budget,benchmark}.md
docs/design/{trust-model,testing}.md; docs/tileops-skills.md
src/tileops/ops/fft.py (public wrapper/validation/cache/dispatch surface only)
workloads/fft.py; tests/ops/test_fft.py
benchmarks/ops/bench_fft.py; benchmarks/benchmark_base.py
src/tileops/manifest/sequence_modeling.yaml structurally via tileops.manifest
round-5 preflight.log and preflight_smoke.py
$HOST_HOME/foreman/local/tileops-container.sh
TileFoundry CLAUDE.md and CLAUDE.local.md
TileFoundry docs/spec/{hir,types,cli,evaluator,runtime}.md and docs/tutorial/*
TileFoundry parser/evaluator/module/tuple/concat/matmul/dtype/runtime-twin sources,
tests, and examples needed for authored-HIR syntax and evaluator behavior
round-3 and round-4 tilefoundry-gaps.md; round-4 authored_hir.py,
first-candidate.md, and unrelated blind TileLang evidence for artifact conventions
unrelated TileLang conventions in mamba/ssd_decode.py and reduction/argreduce.py
targeted rg under src/tileops/kernels and tests with explicit exclusions for
src/tileops/kernels/fft.py and tests/kernels/test_fft_output_layout.py
the newly authored HIR/runtime sources and their generated CUDA/artifacts
```

No forbidden incumbent or output-layout source was read during this interval.
