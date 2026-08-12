# Round 4 Mamba2FwdOp report

## Result

- Current official-stack classification: **measured SOTA** versus the external baseline on CUDA
  13.2, with **no measurable improvement versus the TileOps base incumbent**.
- Branch: `perf/tileops-r4-mamba2`.
- Base: `994f43d374cbc1d77a08e207ab8831402b87a376`.
- Commit: `e987be1ba898c9a7b9ee60420f511eacd0508965`.
- PR: https://github.com/tile-ai/TileOPs/pull/1889 (open; all CI green; GitHub
  `mergeable=MERGEABLE`; awaiting the repository-required one human approval).
- The PR is independently useful as a primary-contract and evidence repair. The original CUDA 12.9
  run did not meet SOTA; the official CUDA 13.2 retest does meet the brief's external-baseline SOTA
  definition without any code change.

The retained implementation dispatches `dt_bias=None` through the existing unbiased
`DaCumsumFwdOp` specialization and caches biased/unbiased variants separately. The PR also fixes
the evidence surface: one workload-owned two-output reference, a literal ordered recurrence,
final-state checks, exact primary rows, and an official wrapper whose FP32 casts remain inside the
timed region. A measured chunk-state configuration change was removed because it regressed the
repeated geometric mean.

## Plan audit

The draft brief was corrected against the repository before implementation:

1. The manifest label `mamba2-2p7b-b1-s2k` says H=80, while the workload model table maps H=80 to
   1.3B and H=128 to 2.7B. The manifest label and exact numeric shape were treated as authoritative;
   the label was not silently rewritten.
2. The existing benchmark always generated and passed `dt_bias`, so it did not measure the named
   primary manifest contract. The focused benchmark now passes no bias or initial state and times
   both outputs.
3. The existing test requested only `y` and compared no final state. The new tests compare both
   FP32 outputs independently.
4. `load_manifest()` already returns the merged op dictionary; no extra `ops` wrapper was assumed.
5. Current TileFoundry uses `tf.reduce`, and `Clamp` requires explicit min/max construction. The HIR
   was authored against those actual surfaces.
6. The documented `--benchmark-json` pytest option is not registered in this checkout. The stale
   command was removed; the native harness writes `profile_run.log`, JUnit properties, and the
   separate focused evidence script writes raw JSON.

## CUDA 13.2 post-rebase authoritative retest

The branch was rebased onto `994f43d374cbc1d77a08e207ab8831402b87a376`. Main changed the
benchmark implementation from sequential `profile()` calls to the shared forward/reverse
`compare()` protocol; the Mamba benchmark conflict retained that new protocol together with this
round's exact two-output primary contract. Because benchmark behavior changed, all claim-bearing
primary measurements were refreshed rather than carried forward.

The common-harness repeated run still used the environment and comparison contract documented
below. Each cell is the median of three rotated trial medians:

| Row | Candidate ms | Base ms | Official 2.3.2.post1 ms | Base/candidate | Official/candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K / H80 / BF16 | 0.723347 | 0.706003 | 0.782755 | 0.9760x | 1.0821x |
| 8K / H64 / FP16 | 0.820499 | 0.824755 | 0.924323 | 1.0052x | 1.1265x |
| geometric mean | 0.770393 | 0.763072 | 0.850599 | 0.9905x | 1.1041x |

Candidate, base, and official trial-median ranges were respectively 0.78% / 26.63% / 2.33% on
the 2K row and 4.15% / 3.11% / 5.36% on the 8K row. Candidate remained faster than official on
every primary row and by 1.1041x in geometric mean, so the environment-specific classification
remains **measured SOTA** against the external baseline. Candidate was 0.96% slower than base in
geometric mean, and the 2K base distribution was highly variable, so there is still no incumbent
improvement or stable per-row regression claim.

Post-rebase correctness/contract results were full Mamba 53/53, benchmark plus manifest contracts
140/140, benchmark smoke 4/4, and exact primary benchmark 2/2. The repeated evidence JUnit has two
passing cases; candidate remained bit-identical to reconstructed base, and errors versus official
were unchanged. Post-rebase JUnit and raw samples are retained separately under
`post-rebase-994f43d/`.

Post-rebase PR CI is fully green. GPU Smoke run `31566456749` / job `94019286214` passed in 8m19s:
the regular smoke scope reported 2915 passed, 14 skipped, and zero failures; the diff-scoped full
Mamba run reported 53 passed and zero failures. Pre-commit, benchmark-contract, compile-contract,
packaging, security-policy, title, gitleaks, and actionlint checks also passed.

## CUDA 13.2 pre-rebase retest

The official runner migration was retested after PR creation, using image
`ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev` at digest
`sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`.
The persistent schema-3 container reported PyTorch 2.13.0+cu132, CUDA 13.2, TileLang
`0.1.11+cu132.gitafcebed1`, `mamba_ssm` 2.3.2.post1, `cupti-python` 13.2.0,
`cuda-bindings` 13.3.1, and an H200 (SM90). No dependency was installed or replaced.

The same common-harness contract was rerun: candidate, reconstructed base incumbent, and official
2.3.2.post1 shared one process, GPU, inputs, rotated trial order, L2/input-shifting policy, and
native CUPTI timing. The official y/final FP32 casts stayed inside timing. Each number below is the
median of three trial medians:

| Row | Candidate ms | Base ms | Official 2.3.2.post1 ms | Base/candidate | Official/candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K / H80 / BF16 | 0.726593 | 0.718752 | 0.777232 | 0.9892x | 1.0697x |
| 8K / H64 / FP16 | 0.817026 | 0.817778 | 0.892851 | 1.0009x | 1.0928x |
| geometric mean | 0.770484 | 0.766668 | 0.833038 | 0.9950x | 1.0812x |

Candidate trial medians were `0.727889, 0.725201, 0.726593 ms` and
`0.819410, 0.817026, 0.815570 ms`, giving only 0.37% and 0.47% run-to-run ranges. It was faster than
official on each row and had a strictly lower geometric mean, so this is **measured SOTA** under the
brief's external-baseline definition. It was 1.08% slower than base on 2K, 0.09% faster on 8K, and
0.50% slower in geometric mean; all are within observed base noise. Therefore it remains a
candidate/base statistical tie, not a TileOps-incumbent performance improvement.

Correctness was unchanged: candidate versus official y/final max errors were
`0.001745611/0.001055131` on BF16 and `0.000199720/0.000082500` on FP16; candidate versus base was
bit-identical. CU 13.2 tests passed: Mamba2 9/9, full Mamba 53/53, manifest 122/122, benchmark unit
22/22, benchmark smoke 4/4, and exact focused benchmark 2/2. All JUnit, logs, raw samples, traces,
environment metadata, and the old/new comparison are in `cu132-retest/`.

The apparent CUDA 12.9-to-13.2 geometric-mean changes were candidate `1.286759 -> 0.770484 ms`
(1.67x), base `1.290544 -> 0.766668 ms` (1.68x), and official `0.902244 -> 0.833038 ms` (1.08x).
These are observations, not stack-only causal estimates: the old report recorded 1500 MHz SM while
the new focused report recorded 1980 MHz, and the host exposes multiple H200s with different idle
clocks. Same-run candidate/base/official comparisons remain valid because all three shared the
same selected device and protocol.

The CU 13.2 profiler retained five TileLang main kernels plus one endpoint copy and the same peak
temporary allocations (88,541,696 / 277,880,832 bytes). Stage times improved broadly: chunk scan
`64.800 -> 54.912 us` and `177.442 -> 154.976 us`; chunk state `24.992 -> 22.144 us` and
`62.593 -> 57.856 us`; total five-main-kernel time `109.856 -> 95.104 us` and
`297.220 -> 265.921 us`. Chunk scan remains dominant.

No CUDA 13.2 compatibility or correctness problem was found, so PR #1889 required no code update.

## Original CUDA 12.9 environment

- Container image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`, brief-pinned digest
  `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`.
- GPU: NVIDIA H200; driver 595.71.05; measured application state reported 1500 MHz SM and 3201 MHz
  memory clocks.
- PyTorch 2.10.0+cu129; CUDA 12.9.
- TileLang `0.1.11+cu129.gitafcebed1`.
- `mamba_ssm` 2.3.1; `cupti-python` 12.8.0.
- TileFoundry reference: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`.
- Every TileOPs Python/GPU command ran through the persistent
  `$HOST_HOME/foreman/local/tileops-container.sh` container. No dependency was installed or
  replaced.

## Blind phase

The blind phase ended at `2026-08-12T04:36:40.176182836+08:00`. Before that timestamp, reads were
limited to the brief-authorized public contract files, installed official wrapper, benchmark
harness, TileFoundry specifications/tutorials/examples, and unrelated repository conventions. No
incumbent sub-op or `src/tileops/kernels/mamba/**` body was read. The detailed read audit is in
`first-candidate.md`.

The sealed candidate originated from `authored_hir.py` and a handwritten TileLang runtime twin:

- TileFoundry `check` passed for a typed FP32-carry one-step graph; `analyze` reported 2,696 FP32
  FLOPs, 11,064 bytes read, and 8,992 bytes written; CTA scheduling returned
  `FEASIBLE_NOT_PROVEN`, 149 ns objective, 34 ops.
- A four-iteration ordered FP32 carry HIR also passed.
- The runtime twin used one CTA per head, a `[P,N]` FP32 shared state, serial token order, immediate
  output, and no full per-token state history or incumbent/external dispatch.
- Small S=512/H=4 FP16 errors versus the literal oracle were y max/mean
  `4.470348e-8 / 2.25549e-9` and state `2.235174e-8 / 1.02261e-9`.
- Full 2K/H80 BF16 errors versus official were y `0.001451045 / 2.699612e-5` and state
  `0.000617206 / 1.716831e-5`.
- Three native-CUPTI medians were 11.194595, 11.197924, and 11.209492 ms. This correct serial
  workaround was about 16x slower than the preflight incumbent and was not proposed for production.

Seal:

```text
first-candidate.md SHA256
8126b78a863b3dd1a394d5ed72d12c78c92e0abafaf8f1ee08bdc3448180577e
```

Primary artifacts: `first-candidate.md`, `first-candidate.sha256`,
`artifacts/blind-first-candidate.{json,xml}`, `authored-hir-{check,analyze,schedule}.json`,
`state-scan-check.json`, and `evidence-sources/mamba2_blind_candidate.generated.cu`.

## Correctness

The final candidate returned exactly the same values as the reconstructed base incumbent for both
outputs. Candidate errors against the contract-equivalent official baseline were:

| Row | y max / mean abs | final state max / mean abs |
| --- | ---: | ---: |
| 2K / H80 / BF16 | 0.001745611 / 0.000044185 | 0.001055131 / 0.000031366 |
| 8K / H64 / FP16 | 0.000199720 / 0.000004476 | 0.000082500 / 0.000003216 |

Both outputs also passed the workload-owned independent SSD factorization on both full rows. Small
S=512 fixtures passed a separate literal FP32 token recurrence across the 255/256 chunk boundary.
Tests assert exact FP32 dtypes and output shapes. Bias, initial-state, default final-state switch,
and same-instance biased/unbiased cache alternation are covered without changing the public API.

## Original CUDA 12.9 performance

Protocol: one process, GPU, input set, precision contract, and persistent container; candidate,
base proxy, and official 2.3.1 were warmed before measurement. Each implementation ran once in each
position of a three-order rotation. Every trial used the repository native-CUPTI `bench_kernel`
harness, including L2 reset/input shifting. The official callable returned
`(native_y.float(), native_final.float())`, with both casts inside the measured call. The base proxy
reinstated exactly the base commit's two changed internal choices: biased dA cumsum and the unchanged
chunk-state default. Raw samples are in `final-profile.json`; `final-profile.xml` has zero failures.

| Row | Candidate trial medians ms | Base trial medians ms | Official trial medians ms |
| --- | --- | --- | --- |
| 2K / H80 / BF16 | 1.236519, 1.229782, 1.228183 | 0.887237, 1.232327, 1.240071 | 0.805572, 0.816453, 0.787844 |
| 8K / H64 / FP16 | 1.349239, 1.346375, 1.342840 | 1.362615, 1.351511, 0.946293 | 1.010517, 1.380168, 0.993637 |

The reported row value is the median of the three trial medians:

| Row | Candidate ms | Base ms | Official ms | Base/candidate | Official/candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K / H80 / BF16 | 1.229782 | 1.232327 | 0.805572 | 1.0021x | 0.6551x |
| 8K / H64 / FP16 | 1.346375 | 1.351511 | 1.010517 | 1.0038x | 0.7505x |
| geometric mean | 1.286759 | 1.290544 | 0.902244 | 1.0029x | 0.7012x |

Candidate trial-median ranges were 0.68% and 0.48%. Base ranges were 28.63% and 30.80%, and official
ranges were 3.55% and 38.25%, due to isolated fast/slow modes. The candidate/base difference is far
below measured noise and is a statistical tie. The candidate is materially slower than official on
the 2K row and has a worse geometric mean. Therefore the original CUDA 12.9 classification was
**no improvement**, not SOTA. It is superseded for the current official stack by the CUDA 13.2
result above.

The PR-focused one-pass run independently agreed on direction: candidate 0.8887/1.3424 ms versus
official 0.8031/1.0165 ms. It is retained in `focused-profile_run.log` and
`focused-benchmark.xml`, but is not substituted for the repeated final result.

## Profiler and implementation evidence

Final one-call PyTorch profiler evidence:

| Row | Peak temporary allocation | chunk scan | chunk state | state passing | CB | dA | endpoint copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K / H80 | 88,541,696 B | 64.800 us | 24.992 us | 11.232 us | 4.704 us | 4.128 us | 2.176 us |
| 8K / H64 | 277,880,832 B | 177.442 us | 62.593 us | 38.081 us | 11.232 us | 7.872 us | 2.560 us |

There are five TileLang main-kernel launches plus one contiguous endpoint-copy launch per call.
The pipeline materializes chunk-local CB and chunk-boundary state, not a full `[B,S,H,P,N]` state
history. Chunk scan is the concrete bottleneck; state passing serializes only across 8 or 32 chunks.
Chunk-state and scan use tensor-core matrix products; state passing uses scalar/vector work. Traces
are `final-mamba2-2p7b-b1-s2k-trace.json` and
`final-mamba2-1p3b-b1-s8k-trace.json`; parsed data is in `final-profiler.log`. Exact blind generated
CUDA and every final/probe source are under `evidence-sources/`.

## Tuning and failed approaches

- Search: five state-passing configs, six chunk-scan configs, four chunk-state configs, each on both
  rows; then isolated native-CUPTI checks of default and three plausible full-pipeline candidates per
  row. Selection was output-gated and then repeated in isolated processes/common harness.
- Approximate resource budget from recorded command wall times: 0.25 single-H200 GPU-hours,
  including about 0.09 compile-hours. No parallel GPU agent or second container was used.
- Repeatedly mutating cached kernel configs in one object produced 0.25-to-0.75 ms drift and was
  rejected as invalid methodology.
- Chunk-scan `threads=64` was slower on both rows. `block_s=128` was slower and introduced only tiny
  within-tolerance y rounding changes. Vectorized state-passing configs were substantially slower.
- Chunk-state `{block_l:64, threads:256}` looked neutral/better in isolated single runs, but the
  common repeated geomean regressed by about 0.28%; it was removed before the PR.
- The unbiased dA dispatch is only 4.1/7.9 us of device work, so removing the unused add cannot move
  the end-to-end result beyond noise.

Ideas adopted after incumbent inspection are explicitly limited to: retaining its existing chunked
SSD factorization and intermediate layout; using its already-supported `has_dt_bias=False` dA kernel
surface; and retaining all incumbent default launch configurations after the alternatives failed.
The blind serial runtime is not relabeled as the production route.

## Tests and artifacts

All commands below ran through `tileops-container.sh`:

| Command | Result | JUnit / artifact |
| --- | --- | --- |
| `python -m pytest -q tests/ops/test_mamba.py -k mamba2` | 9 passed | `required-mamba2.xml`, `required-mamba2.log` |
| `python -m pytest -q tests/ops/test_mamba.py` | 53 passed | `full-test-mamba.xml`, `full-test-mamba.log` |
| `python -m pytest -q tests/test_validate_manifest.py` | 122 passed | `required-manifest.xml`, `required-manifest.log` |
| `python -m pytest -q benchmarks/tests` | 22 passed | `required-benchmark-tests.xml`, `required-benchmark-tests.log` |
| `python -m pytest -q -s benchmarks/ops/bench_mamba2_e2e.py -m smoke` | 4 passed | `required-benchmark-smoke.xml`, `required-benchmark-smoke.log` |
| `python -m pytest -q -s benchmarks/ops/bench_mamba2_e2e.py -m full -k primary_manifest` | 2 passed | `focused-benchmark.xml`, `focused-profile_run.log` |
| `scripts/mamba2_final_evidence.py` | 2 evidence cases, zero failures | `final-profile.{json,xml,log}` |

Test node delta is +3 for `tests/ops/test_mamba.py` (50 to 53) and +4 for the benchmark file (22 to
26): literal final-state/cache coverage plus two small smoke and two exact full benchmark cases.
Ruff, shipped-reference lint, AST compilation, `git diff --check`, and CI pre-commit passed. The
local image and host lacked the `pre_commit` Python package/executable, so dependencies were not
installed; GitHub's pre-commit check supplied the authoritative hook run.

PR CI is fully green. The authoritative CUDA 13.2 GPU Smoke run `31538120566` / job
`93959808269` passed in 8m8s: the regular smoke scope reported 2915 passed, 14 skipped, and zero
failures, while the diff-scoped full Mamba run reported 53 passed and zero failures. Its downloaded
raw logs, generated report, and both JUnit XML files are retained under
`cu132-retest/ci-run-31538120566/`. Pre-commit, compile-contract-gate, benchmark-contract-tests,
packaging, gitleaks, actionlint, title, and security-policy all passed.

## TileFoundry gaps

Detailed structured records and minimal reproducers are in `tilefoundry-gaps.md`.

1. `TF-R4-M2-01`, `semantic-blocker`, duplicate of round 3: dynamic loop-token indexing/output
   insertion cannot express the full ordered scan. `ordered_scan_repro.py` fails with
   `unsupported ShapeDim Var`; `dynamic_scan_output_repro.py` fails because loop `t` resolves to a
   non-Expr `RangeSlice`. Static repeat-interleave head-group expansion works, so this is not the
   round-2 dynamic RepeatInterleave gap. It affects both rows; the serial TileLang workaround costs
   11.197924 ms on the 2K row. Likely ownership is the HIR dynamic indexing/loop-value evaluator.
2. `TF-R4-M2-02`, `ergonomics`, new: canonical source printing emits bare `max_val=inf`. The exact
   authored source passes, but re-import of `authored_hir_roundtrip.py` fails with undefined name
   `inf`. This affects source-to-source round trips for both rows, not runtime correctness. Likely
   ownership is the HIR canonical printer/name environment.

## Remaining risks

- In the historical CUDA 12.9 run, official 2.3.1 was faster and base/official CUPTI trials had
  large bimodal ranges. That result must not be mixed with the current CUDA 13.2 classification.
- The CUDA 13.2 candidate is stable and beats official 2.3.2.post1 on both rows, but it does not beat
  the base incumbent beyond noise. The PR remains primarily a contract/evidence repair.
- `mamba_ssm` is optional in generic installations; full official-comparison tests skip when absent.
  Both pinned evidence environments had their stated official version and executed every case.
- The workload table/manifest model-name mismatch remains a naming issue outside this PR; all
  evidence binds to exact numeric manifest shapes.
- GitHub reports no conflicts and every required check passed, but branch protection requires one
  approving human review. The `tileops-review` team is already requested. The PR was not merged.
- The measured-SOTA label is environment-specific: CUDA 13.2/PyTorch 2.13/official 2.3.2.post1.
  The same code did not beat official 2.3.1 in the original CUDA 12.9 run, and cross-environment
  absolute deltas are confounded by recorded GPU clock differences.
