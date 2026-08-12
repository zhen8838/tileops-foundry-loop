# Five-round preflight

Recorded: 2026-08-11 (Asia/Shanghai)

## Source state

- TileOPs: `5c4d54c44dc60a3bee5bf2b409cf224b7f16c820`
- TileFoundry: `e40f3f666ed95c03a78cae99a54ffb2fc33fed4d`
- TileOPs worktree: clean
- TileFoundry worktree: only the pre-existing untracked `CLAUDE.local.md`

## Runner state

- Image: `ghcr.io/tile-ai/tileops-runner:afcebed1-torch2.10-dev`
- Digest: `sha256:aea905a60995a83438402c9a38a242a3465a18464d3acb11311530c86098754e`
- Docker: 29.1.3, rootless overlayfs, CDI spec
  `$HOST_HOME/.tileops-docker/cdi`
- GPU: NVIDIA H200, compute capability 9.0, 143771 MiB
- Driver: 595.71.05
- Container stack: Python 3.12, PyTorch 2.10.0+cu129, CUDA 12.9,
  TileLang 0.1.11+cu129.gitafcebed1
- External packages: vLLM 0.19.1, flash-linear-attention 0.4.2,
  mamba-ssm 2.3.1, FlashInfer 0.6.11.post2

The host toolkit is CUDA 13.2, but it is not used by the runner. CUDA 12.9 and its `nvcc` are
inside the pinned development image used for every round.

## Actual GPU smoke

Command, pinned to host GPU 1:

```bash
TILEOPS_GPU=1 $HOST_HOME/foreman/local/tileops-container.sh \
  python /opt/tileops-local/tileops-env-smoke.py
```

All calls passed:

- vLLM Triton `fused_experts`
- vLLM Marlin W4A16 `marlin_gemm`
- FLA `chunk_gated_delta_rule`
- mamba_ssm `mamba_chunk_scan_combined`
- `torch.fft.fft` / cuFFT
- TileLang JIT of TileOPs `FFTC2COp`, checked against cuFFT

Raw output: `$TRIAL_SOURCE/preflight/baseline-smoke.log`.

After converting the runner to one persistent container per worktree, the same six calls passed
again inside persistent container `tileops-52b21d40c1de`, pinned to GPU 1. Two separate wrapper
invocations reported the same container hostname and retained a marker in `/tmp`, proving later
commands use `docker exec` rather than `docker run`. Raw output:
`$TRIAL_SOURCE/preflight/baseline-smoke-persistent.log`.

## MoE CUTLASS contract check

The pinned vLLM `cutlass_moe` module exposes FP8/FP4 quantized expert paths, not a BF16/FP16
function matching `FusedMoEExpertsNopadPersistent3WGFwdOp`. The repository benchmark's attempted
imports of `cutlass_moe_fp16` and `cutlass_moe` both fail under vLLM 0.19.1. The fair strongest
runnable same-contract baseline for round 1 is therefore vLLM Triton `fused_experts`.

## Foreman check

The `tileops` project creates worktrees under `$HOST_HOME/TileOPs-worktrees`. A dry-run
confirmed the exact launch is Codex `gpt-5.6-sol` with `model_reasoning_effort=high`, and an
independent detached worktree imported TileOPs from its own `$CONTAINER_WORKSPACE/tileops/src` mount while
seeing the H200 through CDI.

## Post-round-1 shared environment repair

Round 1 exposed two omissions in the initial persistent-container preflight:

- the pinned tag predates the runner Dockerfile layer that installs `cupti-python==12.8.0`, so the
  repository benchmark correctly failed closed instead of using native CUPTI timing;
- a linked worktree's `.git` file points at the main checkout's common Git directory, which was not
  mounted into the container, so history-aware test scripts could not resolve the base commit.

Before round 2, runner environment schema 2 was introduced. Each new per-worktree container now
mounts the TileOPs common Git directory read-only at its original absolute path, sets
`GIT_OPTIONAL_LOCKS=0`, and installs exactly `cupti-python==12.8.0` with `--no-deps`. This matches
`.github/runner/Dockerfile` and does not resolve, upgrade, or replace any pinned CUDA/PyTorch
dependency. The main and completed round-1 containers were recreated once for this schema change;
their worktrees, Git commits, shared `$CI_CACHE`, and round evidence are host-mounted and persisted.

Validation after migration:

- `cupti-python` is 12.8.0, `cuda-bindings` remains 12.9.4, and Torch remains 2.10.0+cu129;
- an actual 1024x1024 FP16 GEMM through `bench_kernel` returned 139 samples with metadata
  `timing=cupti` and no fallback (`cupti-smoke-schema2.log`);
- all six external/TileLang GPU smoke calls passed again (`baseline-smoke-schema2.log`);
- round 1's linked worktree resolved its base commit and `test_node_delta.py` reported the correct
  33 to 34 node delta directly inside the container;
- repeated wrapper calls returned the same main container name, `tileops-52b21d40c1de`.

The smoke helper also now duplicates its original stdout before external kernels initialize. Some
vLLM initialization code replaces process stdout; writing the final JSON report through the saved
descriptor makes the six-case result visible without changing any baseline call or pass/fail rule.
