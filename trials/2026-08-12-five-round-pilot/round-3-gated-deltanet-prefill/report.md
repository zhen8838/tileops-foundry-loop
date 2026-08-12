# Round 3 report: GatedDeltaNetPrefillFwdOp

## Result and provenance

- Classification: **measured SOTA** against the required equivalent FLA 0.4.2
  baseline. Candidate geomean is 2.467790 ms, FLA is 6.486431 ms (2.628437x),
  and no candidate row has a higher three-trial median than FLA.
- Base result: candidate is 2.467790 ms versus base replay 2.487880 ms
  (1.008141x geomean). The useful change is concentrated in the two 4k rows;
  long-context candidate/base differences are measurement noise on the same
  runtime route.
- Branch: `perf/tileops-r3-gated-deltanet-prefill`.
- Base: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`.
- Commit: `9dd66f4fa937d48eb12cbcdf1be53a16877a74f2`.
- PR: https://github.com/tile-ai/TileOPs/pull/1888, open and GitHub-mergeable;
  all CI checks passed; GitHub reports `MERGEABLE` and `BLOCKED` only on the
  repository's human review/merge policy.
- TileFoundry: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
- Image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`, digest
  `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`.
- GPU: NVIDIA H200, UUID
  `GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274`, 143771 MiB, 132 SMs,
  driver 595.71.05.
- Python 3.12.13; PyTorch 2.10.0+cu129; CUDA 12.9; TileLang
  0.1.11+cu129.gitafcebed1; FLA 0.4.2. Raw output is `environment.log`.

The draft brief was checked against the current code before editing. Its
warning about a final-state self-comparison was stale: both layout transition
tests already compare `state_bthd` with `state_bhtd`, so no test was changed
there. The actual benchmark mismatch was stronger than the draft implied:
the incumbent FLA wrapper conditionally omitted `output_final_state`, omitted
`initial_state=None`, and did not cast FLA's FP32 state inside the timed
callable. The PR fixes and tests that contract.

## Blind workflow

The blind phase ended at `2026-08-12T03:13:27.898037498+08:00` before any
incumbent body was read. `first-candidate.md` is mode 0444 with SHA256
`b66da531a6a93b95d71f0895d223a16a767d4091a3729d56874e3cc17b2a279e`.
It contains the full timestamped source-read audit.

`authored_hir.py:GatedDeltaNetStep.delta_step` is a typed one-step graph with
FP32 carry, post-update output, scale 1.0, and the specified recurrence.
`authored-hir-check.json` and `authored-hir-analyze.json` passed; analyze
reported 3,782 FP32 flops and an 8,392-byte HIR peak for its fixed fixture.
`state_scan_hir.py` and `state-scan-check.json` prove a four-step ordered carry
executes. The full scan cannot dynamically insert each readout at its loop
position; the minimal evidence is `dynamic_scan_output_repro.py` and
`dynamic-scan-output-repro.log`.

The independent handwritten TileLang twin used one CTA per head and a 64 KiB
FP32 state without per-token state history. It passed a small FP16 oracle
(output/state max abs 0.000198364/0.000213623) and 4k/H16 BF16
(0.0048828125/0.004638671). Its 4k BF16 native-CUPTI medians were 19.398247,
17.278528, and 20.657372 ms, so it was a correctness bridge, not the shipped
performance path. Its source and runner are retained in `evidence-sources/`.

## Final implementation

After the blind record was locked, incumbent inspection showed a chunked
blocksolve plus fused GDR pipeline, with optional context partitioning. The
PR deliberately adopts that incumbent pipeline and makes no claim that it is
TileFoundry-derived. The new selection does two things:

1. Automatic context partitioning begins at 32k tokens; the explicit force
   override keeps its old behavior. At 4k the same fused GDR runtime executes
   without CP initial-state preparation.
2. The 32k/H16 specialization uses 64 rather than 32 local chunks. Controlled
   FP16/BF16 ablations are in `ablation-32k-h16-*-local64.log`.

No public signature, output shape, dtype, cache key, layout fallback, tune
surface, or kernel-map behavior changed. The base comparator uses the current
body with the base commit's exact route and old local-chunk formula replayed
by environment controls; the kernel diff proves selection is the only runtime
change. Candidate, base, and FLA use the same tensors in one process per
26-row run and all retain both outputs.

## Correctness and test evidence

Every TileOPs Python/GPU command ran through the persistent
`tileops-container.sh` container. Final JUnit XML is under `artifacts/`.

- `python -m pytest -q tests/ops/test_gated_deltanet_prefill.py`:
  15 passed; `gdn-op-final.xml`, `test-gdn-op-final.log`.
- `python -m pytest -q tests/test_validate_manifest.py`:
  122 passed, one existing warning; `validate-manifest.xml`.
- `python -m pytest -q benchmarks/tests`:
  24 passed; `benchmark-tests.xml`. This includes the exact FLA kwargs and
  timed state-cast contract tests.
- `python -m pytest -q -s benchmarks/ops/bench_gated_deltanet_prefill.py`:
  26 passed; `benchmark-op.xml`, `test-benchmark-op.log`, and
  `profile_run.log`.
- `ruff check` on all four changed files and `git diff --check`: passed.
  `pre-commit` is absent from the pinned image and was not installed;
  `pre-commit.log` and `format-tools.log` preserve the check.
- `scripts/test_node_delta.py --base main`: 9 to 15 (+6). Five nodes cover
  distinct partition selector boundaries and one covers the H200 local-chunk
  specialization; see `test-node-delta.log`.

The final 26-row harness first computes FLA, then checks candidate and base
against both FLA outputs at unchanged FP16/BF16 tolerances. The maximum errors
are shown below. All rows report `GatedDeltaNetPrefillFwdKernel`, candidate
route, base route, and allocated-memory deltas in
`artifacts/manifest-profile-final.jsonl` (SHA256
`0150acdf009793d467a816e6e1dc59032a179595f6c56574e17e4913bd3ebb76`).

## Native CUPTI results

Each cell is the median of three trial medians followed by the min/max trial
median. Every trial contains adaptive repeated native-CUPTI samples with
`metadata.timing="cupti"`; all raw samples are in the JSONL. `N/P` means
candidate non-partitioned/base partitioned; all other rows are `P/P`.

| S | H | dtype | route C/B | candidate ms [trial range] | base ms [trial range] | FLA ms [trial range] | C/Base | C/FLA | max abs o/state |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | float16 | N/P | 0.9965 [0.7678, 0.9983] | 1.3098 [0.9471, 1.3101] | 1.2940 [1.2812, 1.2959] | 1.314x | 1.299x | 0.000579834/0.000366211 |
| 4096 | 16 | bfloat16 | N/P | 0.9772 [0.8325, 1.0436] | 1.3123 [1.3101, 1.3186] | 1.3002 [1.2960, 1.3004] | 1.343x | 1.331x | 0.00488281/0.00463867 |
| 32768 | 16 | float16 | P/P | 1.5529 [1.5459, 1.5538] | 1.5518 [1.5482, 1.5524] | 4.3245 [4.3119, 4.3954] | 0.999x | 2.785x | 0.00115967/0.000396729 |
| 32768 | 16 | bfloat16 | P/P | 1.5529 [1.5428, 1.5539] | 1.5451 [1.5358, 1.5620] | 4.3061 [4.2991, 4.3386] | 0.995x | 2.773x | 0.00598145/0.00305176 |
| 65536 | 16 | float16 | P/P | 1.8619 [1.8065, 1.8674] | 1.8203 [1.4283, 1.8232] | 5.3853 [4.2258, 5.4599] | 0.978x | 2.892x | 0.000915527/0.000457764 |
| 65536 | 16 | bfloat16 | P/P | 1.4958 [1.4940, 1.5014] | 1.4986 [1.3860, 1.6626] | 4.2342 [4.1677, 5.3702] | 1.002x | 2.831x | 0.010498/0.00219727 |
| 131072 | 16 | float16 | P/P | 2.4490 [2.3125, 2.4506] | 2.4439 [2.2993, 2.4524] | 7.1992 [7.1753, 7.2107] | 0.998x | 2.940x | 0.00106812/0.000244141 |
| 131072 | 16 | bfloat16 | P/P | 2.3373 [2.3329, 2.4629] | 2.3256 [2.3242, 2.4708] | 7.1537 [7.1370, 7.1633] | 0.995x | 3.061x | 0.0090332/0.00170898 |
| 32768 | 32 | float16 | P/P | 1.8410 [1.8210, 1.8692] | 1.8675 [1.7737, 1.8715] | 4.6307 [4.5496, 4.6778] | 1.014x | 2.515x | 0.00140381/0.000518799 |
| 32768 | 32 | bfloat16 | P/P | 1.7776 [1.7650, 1.8716] | 1.8711 [1.7452, 1.8785] | 4.6740 [4.6532, 4.7486] | 1.053x | 2.629x | 0.00720215/0.00585938 |
| 65536 | 32 | float16 | P/P | 2.3207 [2.0601, 2.4485] | 2.2843 [2.2283, 2.3053] | 5.6530 [5.5063, 5.6858] | 0.984x | 2.436x | 0.000976562/0.000366211 |
| 65536 | 32 | bfloat16 | P/P | 2.3335 [2.1579, 2.4782] | 2.1874 [2.0355, 2.2021] | 5.6536 [5.6505, 5.7129] | 0.937x | 2.423x | 0.00891113/0.00231934 |
| 131072 | 32 | float16 | P/P | 3.6175 [3.2769, 3.6808] | 3.3468 [3.1529, 3.6912] | 10.8329 [10.8321, 10.8360] | 0.925x | 2.995x | 0.0010376/0.000427246 |
| 131072 | 32 | bfloat16 | P/P | 3.3894 [3.1375, 3.7294] | 3.3886 [3.1275, 3.4086] | 10.8446 [10.8419, 10.8494] | 1.000x | 3.200x | 0.00927734/0.00878906 |
| 32768 | 48 | float16 | P/P | 1.9978 [1.9874, 2.0386] | 2.0248 [1.8283, 2.2113] | 5.1975 [5.1865, 5.2143] | 1.014x | 2.602x | 0.00112915/0.000534058 |
| 32768 | 48 | bfloat16 | P/P | 2.1366 [2.0752, 2.2413] | 2.0469 [1.9957, 2.1210] | 5.1764 [5.1004, 5.2433] | 0.958x | 2.423x | 0.00830078/0.00610352 |
| 65536 | 48 | float16 | P/P | 2.8605 [2.6200, 3.0931] | 2.6026 [2.5943, 2.8521] | 7.9897 [7.9775, 7.9977] | 0.910x | 2.793x | 0.00112915/0.000366211 |
| 65536 | 48 | bfloat16 | P/P | 2.9122 [2.9027, 3.1373] | 2.6327 [2.6297, 2.6457] | 8.0093 [8.0052, 8.0104] | 0.904x | 2.750x | 0.00952148/0.00488281 |
| 131072 | 48 | float16 | P/P | 4.9397 [4.5567, 4.9452] | 4.9418 [4.4059, 4.9578] | 15.9422 [15.9307, 15.9459] | 1.000x | 3.227x | 0.00146484/0.000442505 |
| 131072 | 48 | bfloat16 | P/P | 4.9926 [4.4918, 5.0163] | 5.0248 [5.0157, 5.0253] | 15.9822 [15.9772, 15.9969] | 1.006x | 3.201x | 0.0117188/0.00390625 |
| 32768 | 64 | float16 | P/P | 2.3156 [2.1608, 2.4714] | 2.1220 [2.0714, 2.1400] | 5.1371 [4.9230, 5.1947] | 0.916x | 2.218x | 0.00112915/0.000518799 |
| 32768 | 64 | bfloat16 | P/P | 2.1692 [2.0855, 2.1843] | 2.1671 [2.0654, 2.1752] | 5.1405 [4.9365, 5.1651] | 0.999x | 2.370x | 0.0090332/0.00390625 |
| 65536 | 64 | float16 | P/P | 3.4437 [3.0775, 3.6773] | 3.6827 [3.6775, 3.6852] | 9.8250 [9.8130, 9.8278] | 1.069x | 2.853x | 0.00140381/0.000915527 |
| 65536 | 64 | bfloat16 | P/P | 3.6990 [3.1766, 3.7141] | 3.6991 [3.3549, 3.7085] | 9.8501 [9.8487, 9.8505] | 1.000x | 2.663x | 0.00878906/0.00463867 |
| 131072 | 64 | float16 | P/P | 6.0745 [6.0634, 6.0860] | 6.0856 [5.5438, 6.0893] | 19.6310 [19.6275, 19.6552] | 1.002x | 3.232x | 0.00128174/0.000518799 |
| 131072 | 64 | bfloat16 | P/P | 5.7168 [5.4492, 6.1560] | 5.7531 [5.7063, 6.1182] | 19.6747 [19.6720, 19.6814] | 1.006x | 3.442x | 0.0141602/0.00549316 |

The largest within-final-run relative trial-median ranges were 23.12%
(candidate), 27.72% (base), and 28.40% (FLA). Exploratory and final processes
also showed CUPTI multimodality, especially at small shapes; both complete
runs are retained rather than selecting favorable samples. The final run is
the predeclared post-tuning result. Its candidate is below FLA on all 26 row
medians and its 4k candidate trial maxima are below the corresponding FLA
trial minima, while the geomean separation is strict.

## Profiling, memory, and asymptotics

For 4k/H16 BF16, the PyTorch trace records 6 CUDA events and a 27,526,144-byte
peak allocated delta. The timed kernels are cumsum 2.527 us, blocksolve 12.192
us, fused GDR 141.024 us, two fills, and the final state cast. The trace is
`artifacts/profile-4k-h16-final/s4096-h16-bf16-trace.json` (SHA256
`b3e83992e295c78e0045b202d9e77855e41cb7a6f433c99191191ba8e846628c`).

For 131072/H64 BF16, the trace records 16 CUDA events and a 3,294,627,328-byte
peak allocated delta. Dominant kernels are fused GDR 3442.805 us and
blocksolve 1131.430 us; warmup discovery, prepare-h, zero/correct-h0 together
are under 100 us. The trace is
`artifacts/profile-128k-h64-final/s131072-h64-bf16-trace.json` (SHA256
`64c361e19c89a1b091949e97dfe7541b98d648339fddadd68453a324db5511a8`).

Generated CUDA is under each profiler directory. It confirms 512-thread
launch bounds for fused GDR/prepare-h, 32 threads for blocksolve, and the exact
4k non-CP versus 128k CP specializations. NCU counters are unavailable under
host policy (`ERR_NVGPUCTRPERM`), so no unsupported occupancy percentage is
claimed.

The largest candidate allocation delta in the 26-row run is 3,294,626,304
bytes, versus 13,996,392,448 for FLA. At 128k/H64 the two unavoidable large
objects are output `[B,S,H,DV]` (2 GiB BF16) and blocksolve `A[B,S,H,64]`
(1 GiB BF16), plus small CP state/metadata. `output_h=False`, so no
`[B,S/chunk,H,DK,DV]` state history is retained; a forbidden full
`[B,S,H,DK,DV]` FP32 history would be 512 GiB for this row and is absent.

## Tuning budget, failed paths, and risk

The blind runtime used one fixed configuration. Post-blind selection evaluated
two partition choices at 4k and local-chunk values 32/64 at 32k/H16; there was
no autotune or broad search. The first full manifest process ran for 930 s,
including 76 recorded TileLang compilations totaling 591 s (0.1642 compile
hours). The cached final manifest run took 97 s. Including blind checks,
ablations, profilers, and required GPU tests, round GPU occupancy was about
0.40 H200-hours.

Failed or rejected approaches:

- Full authored-HIR output scan is blocked at dynamic loop-indexed insertion.
- The blind serial one-CTA/head runtime is correct but 17-21 ms at 4k.
- Automatically partitioning 4k paid unnecessary CP setup and memory.
- The first 32k/H16 run exposed a noisy FP16 boundary; 64 local chunks gave a
  clear controlled improvement, then the entire distribution was rerun.
- Formatting the entire incumbent kernel with current `ruff format` would
  cause broad unrelated churn; lint and diff checks pass, so it was not done.

Remaining risk is performance variance: identical long-context candidate/base
routes show per-row ratios from 0.904x to 1.069x. The geomean base improvement
is therefore modest and should not be interpreted as a long-context kernel
rewrite. The SOTA claim is specifically against equivalent FLA, where every
row has substantial final-run separation. The 32k/H16 local-chunk rule is
H200-measured; other supported architectures retain correctness but may not
share that tuning optimum.

## TileFoundry gap

`tilefoundry-gaps.md` records `TF-R3-GDN-01` as a new
`semantic-blocker`: a `tile()` induction value becomes a non-Expr
`RangeSlice` when used as an `InsertSlice` offset. It includes the smallest
reproducer, expected/actual behavior, all 26 affected rows, the 17-21 ms
handwritten workaround cost, likely parser/HIR ownership, and comparison with
round 1 `IndexSelect` and round 2 UINT8/dynamic-repeat/i32-schedule gaps.
Carry-only ordered execution is explicitly proven and is not reported as a
bug.

## PR and CI status

PR https://github.com/tile-ai/TileOPs/pull/1888 is open at head
`9dd66f4fa937d48eb12cbcdf1be53a16877a74f2`. GitHub reports
`mergeable=MERGEABLE`, `mergeStateStatus=BLOCKED`, and no review decision. All
non-skipped checks passed: GPU smoke (3m20s), benchmark-contract tests,
compile-contract gate, remote pre-commit, packaging, manifest stats, security,
gitleaks, actionlint, title validation, label, and CI gates. The two skipped
jobs are their workflow's conditional manifest validation/publish steps, not
failures. The remaining block is the repository's human review/merge policy;
the round did not merge the PR.

## CUDA 13.2 retest (2026-08-12)

PR #1888 was retested unchanged at head
`9dd66f4fa937d48eb12cbcdf1be53a16877a74f2` in the official image
`ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`, digest
`sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`.
The environment was Python 3.12.13, PyTorch 2.13.0+cu132, CUDA 13.2,
TileLang 0.1.11+cu132.gitafcebed1, and FLA 0.5.2. The GPU was an NVIDIA
H200, UUID `GPU-777ba55d-d138-d53a-4036-ffeba650c10f`, 143771 MiB, 132
SMs, driver 595.71.05. `cu132-artifacts/logs/cu132-environment.log` is the raw
environment record.

Every GPU/Python command again ran serially through the persistent
`tileops-container.sh` container. The CU132 JUnit evidence is:

- `tests/ops/test_gated_deltanet_prefill.py`: 15 passed in 226.72 s;
  `cu132-artifacts/gdn-op-cu132.xml` and
  `cu132-artifacts/logs/cu132-test-gdn-op.log`.
- `benchmarks/tests/test_gated_deltanet_prefill_contract.py`: 2 passed in
  7.97 s; `cu132-artifacts/gdn-benchmark-contract-cu132.xml` and
  `cu132-artifacts/logs/cu132-test-benchmark-contract.log`.
- `benchmarks/ops/bench_gated_deltanet_prefill.py`: all 26 manifest rows
  passed in 38.42 s with no skip; `cu132-artifacts/gdn-official-benchmark-cu132.xml`,
  `cu132-artifacts/logs/cu132-official-benchmark.log`, and
  `cu132-artifacts/profile_run-cu132.log`.

The same-process candidate/base/FLA evidence harness reran all 26 rows with
three native-CUPTI trials. It kept the original contract: common inputs,
`scale=1.0`, `initial_state=None`, `output_final_state=True`, and FLA's state
cast to the input dtype inside the timed callable. Both outputs were checked
before timing. Candidate/base errors remained identical on every row and all
rows passed the existing FP16/BF16 tolerances. The unmodified raw evidence is
`cu132-artifacts/manifest-profile-cu132.jsonl` (SHA256
`8ff2a8ff4e1bbfc5670eee7c5cd01069124b3fb6a85278cb258087065b3cb999`).

The raw main-matrix geomeans were candidate 2.163234 ms, base 2.080224 ms,
and FLA 4.138249 ms: 0.961627x versus base and 1.912992x versus FLA. One raw
row, `65536,16,bfloat16`, had candidate/base trial medians spanning
1.8668--12.7284 ms and 1.3946--12.7024 ms respectively. Candidate and base
use the same route and produced exactly the same errors, so this is a shared
bimodal timing state rather than a candidate-only regression. A fresh-process
five-trial retest was stable at candidate 1.292546 ms, base 1.299297 ms, and
FLA 2.581795 ms. The raw outlier is preserved; the independent retest is
`cu132-artifacts/retest-64k-h16-bf16.jsonl` (SHA256
`4a6c0852b39d24c5b6aefe01ab7726613b201b958923af866a1038715fd74942`).

The version comparison exposed one further shared slow mode at
`131072,48,bfloat16`: the main process measured candidate/base at
18.0747/18.0630 ms while the paired FP16 row was 3.8337/3.9942 ms. FLA was
also bimodal, with trial medians from 11.1814 to 23.7757 ms. Its independent
five-trial retest was stable at candidate 3.814104 ms, base 3.827240 ms, and
FLA 11.180421 ms, with identical candidate/base errors. That evidence is
`cu132-artifacts/retest-128k-h48-bf16.jsonl` (SHA256
`89bb32a965ec27cefe3d559b7b8845e00cf9a4669cc1e98c0aeb1f246e78c1c5`).

Replacing only those two shared-path bimodal rows with their fresh-process
five-trial medians gives the adjudicated CU132 geomeans: candidate 1.866705
ms, base 1.919125 ms, and FLA 4.006543 ms. This is 1.028081x versus base and
2.146318x versus equivalent FLA; candidate is below FLA on all 26 adjudicated
rows. The original CU129 geomeans were 2.467790/2.487880/6.486431 ms. Thus
the CU132 environment's candidate/base/FLA geomeans are 0.7564/0.7714/0.6177
of the old values. These ratios are whole-environment comparisons, not an
isolated CUDA effect: PyTorch, CUDA, and FLA all changed. The complete
per-row comparison, including raw and adjudicated values, is
`cu132-artifacts/cu132-old-new-comparison.json` and
`cu132-artifacts/cu132-old-new-comparison.md`.

The evidence harness reported candidate route metadata after the base replay,
which leaves `TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS` set. Candidate execution
and timing already clear that variable before every call, so results and
kernel selection were unaffected. `manifest-profile-cu132-corrected.jsonl`
changes only those recorded route fields using the production selector;
`manifest-profile-cu132.jsonl` remains the immutable raw record. This was an
evidence-script defect, not a PR compatibility/correctness issue, so no
TileOPs source change or new commit was made.

The CU132 retest therefore reaffirms the measured-SOTA classification against
the equivalent FLA 0.5.2 contract, with the two post-run noise adjudications
explicitly retained. PR #1888 remains open, `MERGEABLE`, and at the same head;
all existing required checks are successful, and `BLOCKED` continues to mean
only the repository's human review/merge policy.

## Public Foundry Loop PR contract update (2026-08-12)

PR #1888's public title and body were regenerated with the public Foundry Loop
renderer from the round's structured `pr-data.json` and exact `authored_hir.py`.
The title is `[Perf][foundry][LinearAttention] Optimize Gated DeltaNet prefill
selection on H200`. The public body contains only Summary, TileFoundry
Description, Performance, and Result And Limitations in that order. Its table
contains all 26 primary workloads, the candidate, exact incumbent route replay,
and runnable contract-equivalent FLA 0.5.2 baseline using the admitted CU132
evidence. The limitations retain the 13/26 per-row incumbent exceptions, both
shared-path bimodal adjudications, the H200 tuning scope, and the fact that the
shipped selection change is not a direct lowering of the displayed semantic
module. The renderer and public-contract checker passed, and a GitHub API
readback matched the generated title and body byte-for-byte. The PR remains
open and `MERGEABLE` at unchanged head `9dd66f4`; it was not merged.

## Rebase onto main 994f43d (2026-08-12)

Before rebasing, the TileOps worktree was clean and synchronized with the PR
branch at `9dd66f4`. The commit, binary patch, public PR API state, generated
title/body, structured PR data, and authored HIR checksums were preserved. The
branch was then rebased without conflicts onto
`994f43d374cbc1d77a08e207ab8831402b87a376`; `git range-diff` reports the
round patch as equivalent, with new head `d9c8060`.

The new main changes both the GDN benchmark entry and shared timing behavior:
implementations now run through forward/reverse `ManifestBenchmark.compare`,
and CUPTI activities are attributed by per-call timing windows. Performance
was therefore refreshed rather than carrying the prior CU132 table forward.
Post-rebase checks through the persistent official CU132 container passed:
GDN correctness 15/15, the focused FLA contract 2/2, and the official GDN
manifest benchmark 26/26 with no skips. JUnit XML and logs are retained under
`rebase-994f43d/`.

The refreshed evidence harness retained the same inputs, candidate, exact
incumbent route replay, and contract-equivalent FLA 0.5.2 callable, while using
the new main's L2 reset, adaptive repeat, forward/reverse interleaving, and
fail-closed native-CUPTI activity-window attribution. All 26 rows remained
correct with identical candidate/incumbent errors. The full process showed a
shared three-way slow mode only on the 131072-token/64-head FP16 and BF16 rows;
both were rerun in separate fresh processes for five trials, and the raw full
matrix was retained unchanged.

With those two explicit adjudications, candidate/incumbent/FLA geometric means
are 1.892977/1.928791/4.062778 ms. Candidate is 1.0189x faster than incumbent
and 2.1462x faster than FLA geometrically, remains faster than FLA on all 26
primary workloads, and is slower than incumbent on 11 rows. The public
structured data and renderer output were refreshed from these measurements;
the measured-SOTA classification and the limitation that the shipped selection
change is not a direct lowering of the displayed HIR remain unchanged.

The rebased head was pushed with an explicit `--force-with-lease` against the
previous remote head. GitHub readback confirms PR #1888 targets base
`994f43d374cbc1d77a08e207ab8831402b87a376` at head
`d9c80603949599113a1cacfcea0093b92b4c78d7`, and the public title/body match
the generated Foundry Loop files byte-for-byte. All executed CI checks passed,
including `benchmark-contract-tests`, `compile-contract-gate`, and the H200
`gpu-smoke` job (6m18s); the two conditional checks were skipped as designed.
The PR is open and GitHub reports `MERGEABLE`; `BLOCKED` reflects the remaining
human review/merge policy. No merge was performed.
