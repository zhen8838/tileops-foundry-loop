# TileOPs Foundry Loop Playbook

This file is the repository's single normative workflow contract. Plans contain
operator choices, templates contain fillable evidence, and scripts enforce this
contract without redefining it.

## Roles

### Main agent

The main agent is a setup-and-dispatch agent. It does exactly four things:

1. Pin the TileOPs base, the TileFoundry commit, and the official runner image.
2. Build and admit one TileFoundry wheel, then prepare one isolated TileOPs
   worktree and persistent container per round.
3. Create the round state and dispatch exactly one Foreman solo worker per
   operator.
4. Return the task, worktree, container, and pane handles, then stop.

It does not extract the operator contract, choose a kernel design, inspect a
worker's worktree, poll `foreman status`, relay messages, restart workers, watch
CI, edit PRs, or repair TileFoundry. A later human-directed task may collect and
triage findings; it is not part of dispatch.

### Round worker

One worker owns one TileOPs branch through contract discovery, implementation,
measurement, PR, CI, and review follow-up. It does not dispatch another agent.
Its Agent Session starts in its round-state directory, which is the only place
for HIR, runtime twins, profiler scripts, baselines, experiments, and evidence.
The TileOPs worktree is a publication target, not the development workspace. Its
final diff may contain kernel implementation under `src/tileops/kernels/`,
shape-aware production selection under `src/tileops/ops/`, and their necessary
correctness tests. TileFoundry is an installed tool, not a source tree the
worker may inspect or modify.

## Admitted Environment

1. `scripts/build_tilefoundry_wheel.sh` builds a wheel from the exact admitted
   TileFoundry Git commit, outside the checkout, and records the commit and
   SHA-256. Dirty checkout contents cannot enter the wheel.
2. Each persistent TileOPs container installs that wheel and the versioned,
   wheel-only runtime delta in `config/tilefoundry-runtime-requirements.txt`,
   both with `--no-deps`. This fills packages absent from the official runner
   without replacing its CUDA, Torch, TileLang, or baseline stack. The
   TileFoundry checkout is not mounted. The worker uses the installed
   `tilefoundry` command, including `tutorial`, `spec`, `models`, `check`,
   `analyze`, and `schedule`.
3. `scripts/preflight.sh` admits the official runner image, wheel, GPU stack,
   native CUPTI path, and baseline imports before dispatch. Do not install or
   upgrade round dependencies afterward.
4. Foreman's post-worktree hook starts one image-pinned container per worktree.
   Before starting or restarting the Agent Session, Foreman sources the worker
   environment into its pane and adds the repository's `tileops-run` wrapper to
   that session's `PATH`. Prefix unchanged commands with `tileops-run`; it maps
   the pane's current worktree or round directory into that worker's persistent
   container. Put long commands in round-local `evidence/*.sh`; never reconstruct
   the container name, Docker invocation, GPU lock, or absolute container path
   in worker commands. `tileops-run` also supplies the loop and TileOPs Python
   roots; workers must not repeat them with `env PYTHONPATH=...` or `sys.path`.
5. Worktrees and containers isolate code and packages, not GPUs. `tileops-run`
   serializes commands against the physical GPU through the shared `/ci-cache`.
   Native CUPTI attribution fails closed; CUDA-event fallback is diagnostic only.

## Worker Loop

The dispatch prompt stays short because the installed CLI teaches the workflow.
The worker repeats this loop until it reaches the performance target or has an
evidence-bound blocker:

1. Discover the contract from `tileops.manifest`, workload/reference, public Op,
   tests, and benchmark. Do not inspect the incumbent kernel body until the first
   correct production runtime twin is recorded.
2. Ask the installed `tilefoundry` command how to describe and optimize the
   operator. Author the HIR rather than copying a graph supplied by the prompt.
3. Write the actual production TileLang path as an `@runtime_module` twin of the
   HIR and run `tilefoundry check` against it. A detached Torch/evaluator twin
   does not establish provenance.
4. Run `analyze` and `schedule`, change the HIR or implementation in response,
   and repeat. The final HIR must state the chosen topology, mesh/sharding, and
   `gmem` to `smem`/`rmem`/`tmem` movement explicitly.
5. Profile the production path. Use the TileOps kernel corpus and the installed
   TileLang surface to test architecture-appropriate lower-level primitives,
   such as vectorized copies, pipelining/asynchronous movement, explicit layouts,
   warp collectives, MMA/WGMMA, or atomics. The bottleneck decides which
   primitive is relevant; no primitive is mandatory for every operator. While
   the candidate trails the external baseline, at least one profiler-motivated
   lower-level experiment is mandatory.
6. Preserve every TileFoundry limitation as a structured finding with a minimal
   reproducer. Do not edit TileFoundry, open a TileFoundry branch, or propose a
   repair from this round.

## Foundry Provenance Gate

A TileOPs PR may use the `foundry` origin only when all of these are present and
`scripts/check_round.py` passes:

- `authored_hir.py`: the exact final `@module`, with explicit `Mesh` or
  `ShardLayout`, `reshard`, local storage, and return to `gmem`;
- `runtime_twin.py`: an `@runtime_module`/`@runtime_func` wrapper that calls the
  exact production TileLang path measured and proposed in the PR;
- a passing machine-readable `tilefoundry check` report for that twin;
- attempted compute-cost, memory, roofline, timeline, and schedule evidence;
  an unavailable surface needs a matching finding, not a silent omission;
- a decision trace mapping TileFoundry analysis/schedule facts to concrete HIR
  and TileLang choices;
- a profiler-driven lower-level primitive experiment and its measured verdict;
- a substantive base-to-head change to the executed `@T.prim_func`, `@T.macro`,
  or equivalent generated kernel body; and
- a base-to-head diff containing only kernel implementation, shape-aware
  production dispatch, and their correctness tests; any changed public Op
  signature fails the gate; and
- a valid `findings.json`, which may contain an empty list.

Open a performance PR only through `scripts/open_tileops_pr.sh`; it runs this
gate, verifies the production kernel diff, renders the public body, and then
calls GitHub. `render_pr.py` also refuses a scaffolded round whose provenance is
incomplete.

Production dispatch may select kernels, fusion boundaries, and configurations
from contract-owned shapes such as routed rows per local expert, H/F, alignment,
and available parallelism. The unchanged manifest benchmark must reach this
selection through normal Op construction; evaluation code must not pass a
candidate-only switch. Dispatch must preserve the public Op signature, math,
outputs, and supported workload surface. Configuration outside the production
kernel/Op implementation, allocation shortcuts, workload, benchmark, reference,
or manifest changes are forbidden; test-only changes do not qualify. If the
exact production twin cannot pass, the HIR never
reaches an explicit placed form, or no kernel body changed, the round may still
produce useful findings but opens no `[Perf][foundry]` PR.

## Correctness And Performance

Preserve the public Op contract and every supported manifest workload, including
tails and boundary cases. Correctness across that surface is a PR precondition.

Measure candidate, base-commit incumbent, and every runnable same-contract
external baseline in the same container, process, GPU, inputs, precision
contract, warmup policy, timing harness, and implementation-order policy.
Synchronize correctly and exclude compilation, tuning, conversion, and setup
unless the contract includes them. Report every primary workload and repeated-run
noise.

Classify the result as exactly one of:

- `measured SOTA`: no primary row is slower than the fastest runnable external
  baseline beyond measured noise, and candidate geometric-mean latency is lower;
- `improvement without SOTA`: correct and independently useful, but misses that
  condition; or
- `no improvement`: no reviewable performance result, so no performance PR.

## TileFoundry Findings

`findings.json` records only behavior reproduced against the admitted wheel.
Each finding contains a classification (`semantic-blocker`,
`lowering/codegen-blocker`, `runtime-blocker`, `performance-blocker`, or
`ergonomics`), minimal reproducer, expected and actual behavior, affected
workloads, workaround cost, and likely public owning surface. Internal source
paths and speculative fixes are not findings.

Collect findings later with `scripts/collect_findings.py`. Deduplication, priority,
and any TileFoundry repair are a separate human-authorized goal.

## TileOPs PR Contract

The title is:

```text
[<Type>][foundry][<Scope>] <imperative description>
```

The public body contains exactly these sections:

1. `Summary`
2. `TileFoundry Description`: the exact final `@module` class in one Python block,
   with no imports, filenames, paths, or separately printed entrypoint
3. `Performance`: environment, method, and every primary workload. The candidate
   column shows latency only. Each comparator cell has latency on the first line
   and `implementation / candidate` on the second. Incumbent cells are bold. A
   green marker means ratio greater than one and therefore a faster candidate; a
   red marker means ratio at or below one.
4. `Result And Limitations`

Correctness commands, reproducer commands, artifacts, and local paths stay in the
internal report. `pr-data.json` is the sole structured PR input and
`authored_hir.py` is the sole public program source. Generate the body with
`scripts/render_pr.py`; do not hand-edit its table.

## Archive

Use `scripts/archive_trial.py` only after human adjudication. It redacts machine
paths and personal addresses, rejects credential-like content, and omits binary
or duplicate generated state. Historical worker claims remain evidence, not
automatic endorsement.
