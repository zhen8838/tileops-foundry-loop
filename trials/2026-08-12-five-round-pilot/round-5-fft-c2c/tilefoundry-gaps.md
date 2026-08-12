# TileFoundry gaps: round 5 FFTC2COp

TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
The representable f32 real-pair graph is `authored_hir.py`; exact dtype
attempts are retained as three minimal reproducers.

## TF-R5-FFT-01: direct complex tensor dtypes

- Classification: `semantic-blocker`; status: `new`.
- Smallest reproducers: `complex64_dtype_repro.py` and
  `complex128_dtype_repro.py`, each declaring one identity function over a
  tensor with the corresponding public complex dtype.
- Expected: parse and type-check the exact `complex64` or `complex128` tensor
  contract so the authored graph can preserve complex arithmetic and dtype.
- Actual: `tilefoundry check` reports `DType: unknown value 'complex64'` or
  `DType: unknown value 'complex128'`; the emitted valid list contains neither
  complex dtype. Exact outputs are in `complex64-dtype.log` and
  `complex128-dtype.log`.
- Affected rows: both complex64 rows and the complex128 row. An f32 real-pair
  HIR describes the complex64 arithmetic but is not the exact public tensor
  type. Its handwritten TileLang production twin has final native-CUPTI
  medians of 0.0072165 ms (B1) and 0.012096 ms (B64), while direct complex HIR
  remains unavailable.
- Workaround cost: split real/imaginary tensors and explicitly spell complex
  multiply/add. The production wrapper uses a metadata-only interleaved real
  view for contiguous inputs, but the HIR loses direct complex typing and
  cannot itself state the public input/output contract.
- Likely owner: HIR dtype specification/parser, expression type rules, and the
  corresponding evaluator/lowering support. This evidence does not establish
  which downstream layers already support a future complex dtype.

## TF-R5-FFT-02: f64 real-pair precision

- Classification: `semantic-blocker`; status: `new`, related to but distinct
  from `TF-R5-FFT-01`.
- Smallest reproducer: `f64_pair_repro.py`, an identity function over one
  `Tensor[(1, 2), "f64"]`; no complex operator is involved.
- Expected: parse and type-check f64 real tensors so an exact complex128 FFT
  can be expressed as paired real arithmetic.
- Actual: `tilefoundry check` reports `DType: unknown value 'f64'`; see
  `f64-pair.log`. The valid list includes f32 but not f64.
- Affected rows: `fft-4k-c128-b64`. The f32 pair graph is not a
  semantics-preserving substitute. The handwritten TileLang f64 workaround
  passes at `atol=rtol=1e-8`, with max/mean complex absolute error
  `1.994589e-13 / 3.616139e-14`; its final median is 0.015296 ms, 8.43x faster
  than the base incumbent but 1.80x slower than cuFFT.
- Workaround cost: maintain a separate handwritten double-precision TileLang
  path. Runtime precision is preserved, but there is no exact authored-HIR
  provenance for the complex128 row.
- Likely owner: HIR dtype specification/parser and f64 evaluator/lowering
  coverage. Direct complex dtype support would not by itself resolve this
  real-pair reproducer.

## Confirmed non-gaps and environment limits

- Static tuple returns, `tf.concat`, stage layout construction, evaluator,
  canonical source round-trip, `check`, `analyze`, and `schedule` all work for
  the authored f32 butterfly/direct-DFT graph. No dynamic stage index was
  required, so this round does not duplicate rounds 3/4's `RangeSlice` gap.
- cuFFT's specialized one-kernel performance is not a TileFoundry defect.
- NCU counters are unavailable under host policy (`ERR_NVGPUCTRPERM`). This is
  not a TileFoundry defect; evidence uses native CUPTI, PyTorch profiler traces,
  generated CUDA, memory accounting, and controlled ablations.
