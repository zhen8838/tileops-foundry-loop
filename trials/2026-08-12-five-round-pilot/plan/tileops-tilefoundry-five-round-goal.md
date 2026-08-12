# Goal: TileOPs x TileFoundry five-round SOTA kernel loop

You are the single main agent responsible for running this goal end to end. Run five rounds
sequentially. In each round, turn one existing TileOPs operator into a clean TileFoundry problem
description, dispatch exactly one Foreman solo agent using `gpt-5.6-sol` at `high` effort, and
drive the resulting TileLang kernel toward the strongest runnable external baseline on this H200
machine. After all five rounds, consolidate the observed TileFoundry capability gaps and run one
coherent TileFoundry repair batch. Open the relevant PRs and leave reproducible evidence. Do not
use or substitute any GQA operator.

## Fixed environment

- TileFoundry main checkout: `$HOST_HOME/TileFoundry`.
- TileOPs main checkout: `$HOST_HOME/TileOPs`.
- TileOPs GPU command wrapper:
  `$HOST_HOME/foreman/local/tileops-container.sh <command> [args...]`.
- The wrapper maintains one persistent Docker container per TileOPs worktree. The first command
  creates it with `sleep infinity`; later commands execute in the same container via
  `docker exec`. Use `tileops-container.sh shell` for an interactive shell. Keep the container,
  GPU assignment, mounts, installed environment, and caches stable for the entire round; do not
  start a new container per command or create an ad hoc round container.
- Pinned runner image:
  `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`.
- The image is the TileOPs CI stack: CUDA 12.9, Python 3.12, PyTorch 2.10 cu129, the
  matching TileLang commit, vLLM, FlashInfer, FLA, mamba_ssm, and the compiled benchmark
  dependencies. Never upgrade or replace packages in this image during a round.
- The wrapper selects the least-loaded GPU unless `TILEOPS_GPU=<index>` is set, mounts the
  current TileOPs worktree at `$CONTAINER_WORKSPACE/tileops`, mounts TileFoundry read-only at
  `$CONTAINER_WORKSPACE/tilefoundry`, and persists `$CI_CACHE`.
- Host-side TileFoundry commands must use
  `$HOST_HOME/TileFoundry/.venv/bin/<tool>`. Do not rebuild that environment.
- Before round 1, record the TileOPs commit, TileFoundry commit, runner image digest, GPU model,
  driver, CUDA/PyTorch/TileLang versions, and the baseline import/smoke matrix. If this preflight
  differs from the fixed environment or a required baseline is unavailable, repair the shared
  environment first and record the repair; do not silently fall back to a weaker baseline.

## The five rounds

Run this exact order:

1. `FusedMoEExpertsNopadPersistent3WGFwdOp`: strongest runnable same-contract baseline is vLLM
   Triton `fused_experts`. The pinned vLLM 0.19.1 CUTLASS module exposes quantized FP8/FP4
   experts, not the operator's BF16/FP16 contract, so do not compare that quantized path as if it
   were equivalent. Recheck the API during preflight and use CUTLASS only if a genuinely matching
   dtype/math contract is available.
2. `GemmW4A16Op`: strongest runnable baseline is vLLM Marlin W4A16 for `M=1`; retain the
   dequantized PyTorch result only as a correctness oracle, not as the performance target.
3. `GatedDeltaNetPrefillFwdOp`: strongest runnable baseline is FLA
   `chunk_gated_delta_rule` on the production BTHD workloads.
4. `Mamba2FwdOp`: strongest runnable baseline is the official `mamba_ssm` Triton
   `mamba_chunk_scan_combined` implementation.
5. `FFTC2COp`: strongest runnable baseline is `torch.fft.fft`, backed by cuFFT.

An external baseline may be declared unrunnable only with an import/runtime reproducer and the
exact environment facts. A fallback result never counts as SOTA evidence.

## Main-agent loop

For each round, perform all of the following before starting the next round:

1. Read the operator's manifest entry through `tileops.manifest`, its workload/reference, public
   Op signature, correctness tests, and benchmark. Build the brief from those contracts. Do not
   derive the brief by copying the current TileLang kernel body.
2. Write `$TRIAL_SOURCE/round-N-<slug>/brief.md`. It must specify exact
   tensor shapes/layouts/dtypes, math, outputs, numerical tolerances, manifest workloads,
   benchmark commands, external baselines, and the TileFoundry HIR description to implement.
3. Require a two-phase blind evaluation. The solo agent may inspect the manifest, workload,
   reference, Op wrapper, tests, and benchmark, but must not inspect the incumbent TileOPs kernel
   implementation until it has produced and recorded the first correct TileFoundry-derived
   candidate. It may inspect the incumbent afterward for diagnosis and optimization, but the
   report must identify any idea adopted from it.
4. Dispatch from the TileOPs project with this shape, using unique task and branch names:

   ```text
   foreman assign solo --project tileops \
     --prompt "Execute the round brief at <absolute-brief-path>. Take it through a mergeable TileOPs PR and write the required report; do not dispatch another agent." \
     --task <task> --branch perf/<task> \
     --kind codex --model gpt-5.6-sol --effort high
   ```

5. Let the solo agent own implementation, tests, commits, push, PR, CI, and review follow-up.
   Use `foreman status` and the reported pane handle to observe it. Do not edit its worktree or
   relay messages between agents. Do not start the next round until the round has a final report
   and its PR is mergeable, or the report proves why no honest PR can be opened.
6. Require `$TRIAL_SOURCE/round-N-<slug>/report.md` containing:
   commits and PR URL; first-candidate result; final correctness commands; raw per-workload
   latency for candidate, incumbent TileOPs, and every external baseline; speedups; tuning budget;
   profiler evidence; failed approaches; remaining risks; and structured TileFoundry gaps.

## Kernel and performance contract

- The delivered implementation must be a real TileLang kernel produced through the
  TileFoundry description/workflow. It must not call an external baseline or dispatch to the
  incumbent implementation.
- Preserve the public TileOPs Op contract and manifest-owned workloads. Correctness must pass on
  every supported manifest dtype/shape, including boundary/tail cases, before performance is
  claimed.
- Benchmark candidate, incumbent, and external baselines in the same container, process, GPU,
  inputs, precision contract, warmup policy, and timing harness. Exclude compilation and tuning
  from steady-state latency, synchronize correctly, and retain raw output plus `profile_run.log`.
- Tune against the full manifest distribution rather than one cherry-picked shape. Do not remove
  slow rows, loosen tolerances, change layouts, precompute runtime work, or include less work in
  the candidate timing.
- The SOTA target is: on every primary manifest workload, the final candidate is no slower than
  the fastest runnable external baseline within benchmark noise, and its geometric-mean latency
  is strictly lower. Also report comparison with the incumbent TileOPs kernel. Use repeated runs
  to quantify noise; never label a regression or statistically indistinguishable result SOTA.
- If the target is not reached after substantive profiling and optimization, preserve the best
  correct candidate only when it is independently useful and reviewable. Otherwise open no
  performance PR. In both cases, finish the report with measured evidence and concrete blockers.

## TileFoundry gap protocol

Every suspected gap must be demonstrated against the current TileFoundry checkout and classified
as `semantic-blocker`, `lowering/codegen-blocker`, `runtime-blocker`, `performance-blocker`, or
`ergonomics`. Record:

- the smallest HIR/TIR reproducer or generated description that fails;
- expected behavior and actual error/output;
- affected round/workloads and measured workaround cost;
- likely owning spec/module, without prescribing an unverified implementation;
- whether the gap is new, duplicate, or an enhancement rather than missing support.

Workarounds are allowed for a round only when they preserve semantics and timing fairness. Do not
call every optimization opportunity a TileFoundry bug.

After round 5, deduplicate the five reports into
`$TRIAL_SOURCE/tilefoundry-gaps.md`. Select one coherent repair batch that
unblocks the largest number of rounds without combining unrelated architecture changes. Write a
normal finalized TileFoundry plan under `docs/plans/` using the repository template and policy
preflight, then dispatch exactly one repair agent:

```text
foreman assign solo --project tilefoundry --plan <relative-plan-path> \
  --task tileops-gap-repair --branch feat/tileops-gap-repair \
  --kind codex --model gpt-5.6-sol --effort high
```

The repair agent owns the TileFoundry PR through tests, CI, and review. Re-run the minimal gap
reproducers and the affected TileOPs candidate benchmarks after the repair; report before/after
results. Leave deferred gaps in the consolidated document with evidence and a proposed priority.

## Done

This goal is complete only when all five rounds have reports, every successful kernel has a
mergeable TileOPs PR, unsuccessful rounds have honest reproducible blocker evidence, the gap
inventory is deduplicated, one coherent TileFoundry repair PR is mergeable (or the inventory
proves there was no valid repair candidate), and a final summary links every brief, report, PR,
benchmark artifact, fixed gap, and deferred gap. The final summary must distinguish measured
SOTA, improvement without SOTA, and no improvement. Never merge PRs yourself.
