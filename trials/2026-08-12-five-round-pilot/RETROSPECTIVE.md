# Pilot Retrospective

Human adjudication: 2026-08-12.

## Accepted Outcome

Only round 2, [TileOPs PR #1887](https://github.com/tile-ai/TileOPs/pull/1887),
is retained as an acceptable kernel result from this pilot. It adds a new
`GemmW4A16DecodeKernel` with a substantive `@T.prim_func main` body and routes
the `M=1` contract to it. It improves over the TileOPs incumbent but does not
beat Marlin, so its honest classification remains `improvement without SOTA`.

## Rejected Performance PRs

| Round | PR | Human decision | Kernel-body audit |
| ---: | --- | --- | --- |
| 1 | [#1886](https://github.com/tile-ai/TileOPs/pull/1886) | Reject as a TileFoundry performance-kernel result | Changes a sparse config, default selection, and output allocation. It does not change an executed `@T.prim_func` or `@T.macro` body. |
| 3 | [#1888](https://github.com/tile-ai/TileOPs/pull/1888) | Reject as a TileFoundry performance-kernel result | Changes partition thresholds and routing helpers. It does not change an executed `@T.prim_func` or `@T.macro` body. |
| 4 | [#1889](https://github.com/tile-ai/TileOPs/pull/1889) | Reject as a TileFoundry performance-kernel result | Changes operator, workload, benchmark, and correctness evidence only; no TileLang kernel file is changed. |
| 5 | [#1892](https://github.com/tile-ai/TileOPs/pull/1892) | Do not retain from this pilot | This PR does contain an FFT kernel-body change, so it must not be conflated with rounds 1, 3, and 4. The human decision was nevertheless that only #1887 was acceptable from this run; the archived evidence remains available for separate analysis. |

The original reports' performance classifications describe their measured
comparisons, but performance alone does not satisfy the loop's provenance and
implementation objective. In particular, #1888 reporting SOTA against FLA
cannot convert a dispatch/selection-only change into a generated TileFoundry
kernel.

## Root Cause

The prompt required a real TileLang kernel, but completion was gated mostly on
correctness, benchmark results, and PR mergeability. Agents could therefore
optimize an existing implementation through configuration, routing, allocation,
or wrapper changes and still present the outcome as a TileFoundry-originated
kernel result. Human review happened after PR creation, too late to prevent
invalid PRs.

## Loop Changes

1. A `[Perf][foundry]` PR now requires a substantive base-to-head change in an
   executed `@T.prim_func`, `@T.macro`, or equivalent generated kernel body.
2. Dispatch, wrapper, allocation, configuration, workload, benchmark, and test
   changes do not satisfy that requirement on their own, even inside a kernel
   source file.
3. `scripts/check_kernel_diff.py` enforces the structural gate before a PR is
   opened. The report must name the changed symbol and connect it to the authored
   HIR and measured route.
4. A missing kernel-body diff ends the round as `no improvement`; it produces a
   failure report and no performance PR.
5. Every multi-round run is archived with `scripts/archive_trial.py`, including
   failed rounds and a human retrospective, so later analysis does not depend on
   agent self-classification.

## Worker Shutdown

All five pilot Foreman workers were stopped after adjudication. Their worktrees
and branches were retained locally for analysis; no PR was merged by the loop.
