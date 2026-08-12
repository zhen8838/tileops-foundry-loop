# 2026-08-12 Five-Round Pilot

This directory is the sanitized analysis archive for the first five-round
TileOPs x TileFoundry pilot. Start with [RETROSPECTIVE.md](RETROSPECTIVE.md): it
contains the human-adjudicated outcome and supersedes agent self-classification
for loop acceptance.

The original agent-authored [final summary](final-summary.md), briefs, reports,
HIR programs, PR data, benchmark output, profiler traces, failed attempts, and
gap records are retained as evidence. Their claims are historical inputs, not
an endorsement by the loop maintainer.

[MANIFEST.md](MANIFEST.md) records every copied and omitted file. Host paths and
personal email addresses were redacted. Binary tensors, caches, Git bundles,
and duplicate CI/worktree downloads were omitted; the original local state was
left unchanged.

## Navigation

| Round | Operator | Brief | Report | PR |
| ---: | --- | --- | --- | --- |
| 1 | `FusedMoEExpertsNopadPersistent3WGFwdOp` | [brief](round-1-fused-moe-experts/brief.md) | [report](round-1-fused-moe-experts/report.md) | [#1886](https://github.com/tile-ai/TileOPs/pull/1886) |
| 2 | `GemmW4A16Op` | [brief](round-2-gemm-w4a16/brief.md) | [report](round-2-gemm-w4a16/report.md) | [#1887](https://github.com/tile-ai/TileOPs/pull/1887) |
| 3 | `GatedDeltaNetPrefillFwdOp` | [brief](round-3-gated-deltanet-prefill/brief.md) | [report](round-3-gated-deltanet-prefill/report.md) | [#1888](https://github.com/tile-ai/TileOPs/pull/1888) |
| 4 | `Mamba2FwdOp` | [brief](round-4-mamba2/brief.md) | [report](round-4-mamba2/report.md) | [#1889](https://github.com/tile-ai/TileOPs/pull/1889) |
| 5 | `FFTC2COp` | [brief](round-5-fft-c2c/brief.md) | [report](round-5-fft-c2c/report.md) | [#1892](https://github.com/tile-ai/TileOPs/pull/1892) |

The consolidated TileFoundry gaps are in
[tilefoundry-gaps.md](tilefoundry-gaps.md), with repair follow-up in
[post-repair/report.md](post-repair/report.md).
