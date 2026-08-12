# TileOPs x TileFoundry five-round final summary

Completed: 2026-08-12 (Asia/Shanghai)

## Outcome

Five non-GQA operators were run sequentially with exactly one Foreman solo
agent per round (`gpt-5.6-sol`, high effort). All five agents produced a final
report and an open TileOPs PR. One round met the stated external-SOTA contract,
three produced useful incumbent improvements without external SOTA, and one
was a statistical tie with the incumbent and remained behind its external
baseline. No PR was merged.

| Round | Operator | Classification | Result | PR |
| ---: | --- | --- | --- | --- |
| 1 | `FusedMoEExpertsNopadPersistent3WGFwdOp` | improvement without SOTA | GM 5.4219 ms vs incumbent 5.4830 and vLLM 5.7403; failed the every-row rule | [#1886](https://github.com/tile-ai/TileOPs/pull/1886) |
| 2 | `GemmW4A16Op` | improvement without SOTA | 1.680x GM speedup over incumbent; 3.29x slower than Marlin | [#1887](https://github.com/tile-ai/TileOPs/pull/1887) |
| 3 | `GatedDeltaNetPrefillFwdOp` | **measured SOTA** | faster than FLA on all 26 primary rows; final production route honestly records incumbent-derived ideas | [#1888](https://github.com/tile-ai/TileOPs/pull/1888) |
| 4 | `Mamba2FwdOp` | no improvement | GM 1.286759 ms vs incumbent 1.290544 (noise tie), official mamba_ssm 0.902244 | [#1889](https://github.com/tile-ai/TileOPs/pull/1889) |
| 5 | `FFTC2COp` | improvement without SOTA | 10.6528x GM speedup over incumbent; 1.7554x slower than cuFFT | [#1892](https://github.com/tile-ai/TileOPs/pull/1892) |

## Round evidence

| Round | Brief | Report | Primary raw benchmark evidence |
| ---: | --- | --- | --- |
| 1 | [brief](round-1-fused-moe-experts/brief.md) | [report](round-1-fused-moe-experts/report.md) | [profile](round-1-fused-moe-experts/evidence/profile_run.log), [full benchmark](round-1-fused-moe-experts/evidence/benchmark-full.log) |
| 2 | [brief](round-2-gemm-w4a16/brief.md) | [report](round-2-gemm-w4a16/report.md) | [native profile](round-2-gemm-w4a16/profile-candidate.log), [benchmark artifact directory](round-2-gemm-w4a16/local-artifacts/) |
| 3 | [brief](round-3-gated-deltanet-prefill/brief.md) | [report](round-3-gated-deltanet-prefill/report.md) | [final JSONL](round-3-gated-deltanet-prefill/artifacts/manifest-profile-final.jsonl), [profile](round-3-gated-deltanet-prefill/profile_run.log) |
| 4 | [brief](round-4-mamba2/brief.md) | [report](round-4-mamba2/report.md) | [repeated profile](round-4-mamba2/final-profile.json), [focused profile](round-4-mamba2/focused-profile_run.log) |
| 5 | [brief](round-5-fft-c2c/brief.md) | [report](round-5-fft-c2c/report.md) | [interleaved profile](round-5-fft-c2c/worktree-artifacts/interleaved-profile_run.log), [required profile](round-5-fft-c2c/worktree-artifacts/required-profile_run.log) |

Each report includes the blind first candidate, correctness commands, raw
candidate/incumbent/external latency, tuning budget, profiler evidence, failed
approaches, risks, commit, CI, and TileFoundry gaps.

Final GitHub audit reports all five TileOPs PR heads as `MERGEABLE`. Their
first complete GPU Smoke attempts passed. A later batch rerun for rounds 1, 2,
4, and 5 is currently queued behind both repository H200 runners, which are
occupied by the CUDA 13.2 runner-image migration; this makes GitHub's aggregate
`mergeStateStatus` read `BLOCKED` until those duplicate checks finish. Round 3
had one transient actionlint download failure (`curl` rejected a self-signed
certificate); rerunning only that failed job passed. No round has a current
code/test failure, and human review remains pending where branch policy
requires it.

## Fixed environment

Preflight is in [preflight/README.md](preflight/README.md), with raw six-baseline
smoke in [baseline-smoke-persistent.log](preflight/baseline-smoke-persistent.log).
The fixed stack was TileOPs `5c4d54c`, TileFoundry `e40f3f6`, H200, driver
595.71.05, CUDA 12.9, Python 3.12, PyTorch 2.10.0+cu129, TileLang 0.1.11,
vLLM 0.19.1, FLA 0.4.2, mamba_ssm 2.3.1, and FlashInfer 0.6.11.post2 in image
digest `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`.

The wrapper used one persistent container per worktree. The first call created
it; all later commands used `docker exec`, retaining `$CI_CACHE`, installed
runner-only CUPTI support, and the worktree mount. Preflight proves reuse with
the same container hostname and a persistent `/tmp` marker across invocations.

## TileFoundry repair

The [deduplicated gap inventory](tilefoundry-gaps.md) contains eight unique
gaps. The only duplicated blocker, `TF-X-SCAN-01`, affected rounds 3 and 4 and
was selected as one coherent repair batch. The finalized plan is
`$HOST_HOME/TileFoundry/docs/plans/tileops-scan-window-indexing/PLAN.md`.

TileFoundry [PR #90](https://github.com/tile-ai/TileFoundry/pull/90) at
`e0c46ff111bfbb0cb64b21b6209efe6979b7a6a8` fixes absolute dynamic tile-window
coordinates across parser, HIR, evaluator, HIR-to-TIR, and CUDA codegen. Its CI
passed. Source validation was 919 passed/1 skipped, installed and source blast
radius were each 52/52, and all four round-3/round-4 minimal/full authored HIR
reproducers passed after the repair.

The [post-repair report](post-repair/report.md) records the independent reruns.
R3 passed 26/26 benchmark rows and retained its SOTA direction. R4 passed both
primary rows and retained its no-improvement direction. Seven unrelated gaps
remain deferred with reproducer/effect/priority in the inventory: vector
`IndexSelect`, `u8`, symbolic `RepeatInterleave`, dense i32 H200 scheduling,
infinity printing, complex dtypes, and `f64`.

## Goal prompt

The exact reusable one-goal prompt is retained at
[plan/tileops-tilefoundry-five-round-goal.md](plan/tileops-tilefoundry-five-round-goal.md).
