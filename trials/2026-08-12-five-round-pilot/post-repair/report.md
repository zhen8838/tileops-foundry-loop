# TileFoundry repair verification

Verified: 2026-08-12 (Asia/Shanghai)

## Repair under test

- Branch: `feat/tileops-gap-repair`
- Commit: `e0c46ff111bfbb0cb64b21b6209efe6979b7a6a8`
- PR: https://github.com/tile-ai/TileFoundry/pull/90
- CI: https://github.com/tile-ai/TileFoundry/actions/runs/31548136051

The repair makes two-argument `tile(extent, step)` windows use one absolute
element-start coordinate through parser binding, Slice inference, evaluation,
HIR-to-TIR, and CUDA TensorView lowering. The PR's final validation recorded
919 passed/1 skipped source tests, 52/52 installed tests, 52/52 source
blast-radius tests, and 4/4 golden reproducers.

## Independent authored-HIR rerun

Using the repair worktree's installed `tilefoundry` CLI:

- Round 3 minimal `ScanCopy`: PASS.
- Round 4 minimal `ScanCopy`: PASS.
- Round 3 complete `GatedDeltaNetStep.delta_step`: both outputs PASS `nan_inf`.
- Round 4 complete `Mamba2Step.mamba2_step`: both outputs PASS `nan_inf`.

Two initial CLI calls named nonexistent selectors and exited before parsing the
program. They were invocation errors, not product failures; reruns used the
class/function selectors declared by the source files above.

## Affected TileOPs rerun

Both commands ran through the per-worktree persistent container wrapper, which
uses `docker exec` after the first invocation:

```bash
# Round 3
python -m pytest -q -s benchmarks/ops/bench_gated_deltanet_prefill.py
# 26 passed in 36.78s

# Round 4
python -m pytest -q -s benchmarks/ops/bench_mamba2_e2e.py -m full -k primary_manifest
# 2 passed, 24 deselected in 16.97s
```

Round 3 remained faster than FLA on all 26 rows. Round 4 remained slower than
official mamba_ssm on both rows (candidate 0.8952/1.3559 ms, external
0.8196/1.2053 ms). The handwritten TileLang candidates do not consume the
TileFoundry implementation at runtime, so unchanged direction is expected.

The post-repair commands reused the same per-worktree containers and their
compile caches; the wrapper did not create a container per command.

- `round-3-profile_run.log`, SHA256
  `4cf15bb9e8f08dd87558cad9aab578bb8442292ee207ed0da04e54be2ce2006b`
- `round-4-profile_run.log`, SHA256
  `2f070dbe5733b6b8bfad6bfedc93cc1fcaad69ba9d5325af88a95527cc57727f`
