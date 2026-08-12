# Round report: {{OPERATOR}}

Completion is governed by the provenance and PR gates in `PLAYBOOK.md`.

## Result

- Classification: `measured SOTA | improvement without SOTA | no improvement`
- Branch, commits, and PR:
- Candidate vs incumbent geometric mean:
- Candidate vs strongest external geometric mean:

## Environment

- TileOPs base/head:
- TileFoundry wheel commit/version/SHA-256:
- Image tag/digest, GPU, driver, CUDA, PyTorch, TileLang, CUPTI:

## Contract

- Manifest, workload/reference, public Op, tests, and benchmark read:
- Inputs, outputs, layouts, dtypes, math, mutation, and tolerances:

## TileFoundry Provenance

- Final placed HIR and selector:
- Production `@runtime_module` twin and exact TileLang route:
- `check` report:
- Analyze/schedule reports:
- Decision trace:

## Kernel And Primitive Work

- Changed `@T.prim_func`/`@T.macro`:
- Profiler bottleneck:
- Lower-level primitive experiments and measured verdicts:
- Tuning budget and failed candidates:

## Correctness And Performance

- Commands, tolerances, and raw evidence:
- Per-workload candidate/incumbent/external samples and noise:
- Geometric means and SOTA condition:

## Findings

- `findings.json`:
- Minimal reproducers:

## Risks And PR Status

- Residual risk:
- CI, review, and mergeability:
