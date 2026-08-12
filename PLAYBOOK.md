# TileOPs Foundry Loop Playbook

This file is the single normative contract for the repository. Concrete plans,
round briefs, reports, scripts, and examples may reference it; they must not
restate or override it.

## Roles

- The main agent selects operators, extracts public contracts, admits the
  environment, orders rounds, consolidates gaps, and classifies results.
- Exactly one solo agent owns each TileOPs round from implementation through
  tests, commit, push, PR, CI, and review follow-up. It must not dispatch
  another agent.
- Retest and human-review work returns to the same round agent and worktree.
- Agents open and maintain PRs but never merge them.

## Environment

1. Set `TILEOPS_FOUNDRY_LOOP_ROOT` to this checkout and source `.env`.
2. Run `scripts/preflight.sh` before the first round and whenever the runner
   image digest changes. Do not install or upgrade packages in an admitted
   round environment.
3. Run TileOPs Python and GPU commands through
   `scripts/tileops-container.sh`. Each TileOPs worktree owns one persistent,
   image-pinned container.
4. Worktrees and containers isolate source and editable installs, not GPUs.
   Wrap every performance run, and correctness runs that compile many kernels,
   with `scripts/with_gpu_lock.sh <gpu> ...`.
5. An image change creates a new environment schema. Recreate stale
   containers, rerun stack/baseline/CUPTI admission, and append retest evidence
   without overwriting old measurements.

Native CUPTI attribution is required for claim-bearing measurements and fails
closed. CUDA-event fallback is diagnostic only and cannot support a SOTA or
performance claim.

## Round Protocol

### Contract extraction

The main agent reads the operator through `tileops.manifest`, then its
workload/reference, public Op wrapper, correctness tests, and benchmark. The
round brief records exact inputs, outputs, shapes, layouts, dtypes, mutation,
math, accumulation/rounding, tolerances, dispatch/fallback behavior, primary
workloads, strongest same-contract external baseline, and evaluation commands.

Do not derive a brief from the incumbent TileLang kernel body.

Create a round with:

```bash
uv run python scripts/new_round.py \
  --slug <slug> --scope <TileOPs-scope> --operator <OpClass>
```

### Blind gate

Before recording the first correct TileFoundry-derived candidate, the round
agent may inspect only the contract sources above, TileFoundry specifications
and examples, and unrelated TileLang examples. It must not read, search,
import-inspect, disassemble, or otherwise expose the incumbent kernel body.

Record an immutable `first-candidate.md` containing the source audit, authored
HIR, correctness, latency, and limitations. Incumbent inspection is allowed
only afterward for diagnosis and optimization. The report identifies every
idea adopted from it; an incumbent-derived route must never be presented as
TileFoundry-generated.

### Implementation and evidence

- Deliver a real TileLang kernel produced through the TileFoundry workflow. It
  must not call an external baseline or dispatch to the incumbent.
- A `[Perf][foundry]` PR must contain a substantive change to the executed
  TileLang kernel body produced in the current round. Changes limited to
  dispatch, operator wrappers, allocation, configuration constants/defaults,
  workload selection, benchmarks, or tests do not qualify, even when they live
  in a file under `src/tileops/kernels/`.
- Before opening a performance PR, record the base-to-head kernel diff and name
  the changed `@T.prim_func`, macro, or equivalent generated kernel body in the
  round report. If that evidence is absent, classify the round as `no
  improvement`, open no performance PR, and retain the analysis as a failed
  round. Do not relabel incumbent selection or orchestration as a
  TileFoundry-generated kernel.
- Enforce that gate before push with:

  ```bash
  uv run python scripts/check_kernel_diff.py \
    --repo "$TILEOPS_ROOT" --base <admitted-base> --head HEAD \
    --kernel <relative-kernel-path>:<prim-func-or-macro-name>
  ```

  A passing check is necessary, not sufficient: the report must still connect
  the changed body to the authored HIR and the measured candidate route.
- Preserve the public Op contract and every supported manifest workload,
  including boundary and tail cases.
- Correctness across the supported surface is a hard precondition for a PR.
- Keep exact commands, tolerances, raw logs/JUnit, profiler output,
  `profile_run.log`, tuning budget, failed approaches, residual risks, commits,
  PR state, and CI state in the internal round report and artifact directory.

### Performance contract

Measure candidate, incumbent, and every runnable same-contract external
baseline in the same container, process, GPU, inputs, precision contract,
warmup policy, timing harness, and implementation order policy. Synchronize
correctly and exclude compilation/tuning from steady-state latency.

Report every primary manifest workload. Do not remove slow rows, loosen
tolerances, change layouts, precompute runtime work, or time less work. Repeat
runs to quantify noise.

Classify the result as exactly one of:

- `measured SOTA`: candidate is no slower than the fastest runnable external
  baseline within measured noise on every primary workload, and has strictly
  lower geometric-mean latency.
- `improvement without SOTA`: correct and independently useful, but misses the
  SOTA condition.
- `no improvement`: no independently useful performance result; open no
  performance PR and preserve reproducible blocker evidence instead.

An external baseline is unrunnable only with an import/runtime reproducer and
exact environment facts. A fallback never counts as SOTA evidence.

## TileFoundry Gaps

Record only demonstrated current-checkout capability gaps. For each gap keep a
minimal HIR/TIR/CLI reproducer, expected and actual behavior, affected
workloads, measured workaround cost, likely owning module, and whether it is
new, duplicate, or an enhancement. Classify it as one of:

`semantic-blocker`, `lowering/codegen-blocker`, `runtime-blocker`,
`performance-blocker`, or `ergonomics`.

A valid semantic and fair workaround is allowed. An unimplemented optimization
idea is not automatically a TileFoundry bug.

## TileOPs PR Contract

The title is:

```text
[<Type>][foundry][<Scope>] <imperative description>
```

`<Type>` is the honest TileOPs change type (`Perf` for a delivered performance
kernel, `Fix` for a correctness fix, and so on). `foundry` is the origin slot
introduced by merged TileOPs PR
[#1894](https://github.com/tile-ai/TileOPs/pull/1894), and `<Scope>` is the
operator family such as `GEMM`, `MoE`, `Mamba`, or `FFT`. Use the origin only
for a genuinely TileFoundry-generated change.

The public body contains exactly these sections in order:

1. `Summary`
2. `TileFoundry Description`: one Python block containing the exact `@module`
   class, with no imports, filename, path, or separately printed entrypoint
3. `Performance`: environment, method, and every primary workload against the
   candidate, incumbent, and every runnable external baseline. Each
   candidate column shows latency only. Every comparator column combines
   latency and `implementation / candidate` on two lines, with the marker and
   ratio kept on one non-breaking line. The incumbent cells are bold. A red
   marker denotes a ratio above one (candidate speedup); a green marker denotes
   a ratio at or below one.
4. `Result And Limitations`: classification, per-row exceptions, noise, and
   remaining limitations

Correctness commands, reproduction commands, artifacts, filenames, and local
paths are internal evidence and must not appear in the public body.

`pr-data.json` is the sole structured PR input. `authored_hir.py` is the sole
program source; the renderer extracts its unique top-level `@module` class and
strips file-level imports. Generate and validate instead of hand-editing
tables:

```bash
uv run python scripts/render_pr.py rounds/<slug>/pr-data.json \
  --output-dir rounds/<slug>
uv run python scripts/check_pr.py rounds/<slug>/pr-data.json
```

The renderer owns the input schema, section layout, ratios, geometric means,
and leak checks. `templates/pr-data.json` is the fillable starter; the rendered
example under `examples/` is the visual reference.

## Trial Archive

At the end of a multi-round attempt, import the round state into a dated
directory under `trials/` with `scripts/archive_trial.py`. Preserve briefs,
reports, authored HIR, structured PR data, benchmark/profiler evidence,
reproducers, and failure records. The archive command must redact host paths
and personal email addresses, reject credential-like content, and omit binary
tensors, caches, Git bundles, and other non-reviewable generated state. Add a
retrospective that distinguishes kernel-generation failures from correct
kernels that missed the performance or review bar.

## Review and Retest

For human review, restate each comment as a concrete contract, correctness,
performance, or presentation requirement. Reproduce technical claims, make the
smallest coherent change, rerun affected correctness and the full primary
performance distribution when runtime behavior changes, update internal
evidence and `pr-data.json`, regenerate the body, reply with evidence, and own
CI until the PR is mergeable.

For a runner-image retest, rebase when the old base has stale stack assertions;
record the new image digest and full stack; rerun official admission,
PR-relevant correctness, and the same comparison contract; compare every
workload with old evidence; attribute deltas only when isolated; update code
only for a genuine compatibility, correctness, or review issue.

## Multi-Round Completion

Run rounds sequentially unless the concrete plan says otherwise. After all
rounds, deduplicate demonstrated gaps and select at most one coherent
TileFoundry repair batch that unlocks the largest useful surface. The repair
agent owns its PR through tests, CI, and review. Rerun minimal reproducers and
affected TileOPs benchmarks after repair, and distinguish measured SOTA,
improvement without SOTA, and no improvement in the final summary.
