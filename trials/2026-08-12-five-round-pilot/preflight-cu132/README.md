# CUDA 13.2 runner migration preflight

Recorded: 2026-08-12 (Asia/Shanghai)

## Official image

- Tag: `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`
- Digest: `sha256:2590888968a870216e1ea076829b909e62e9053608f6a65d5ab5ecd7eb5561f7`
- Source recipe: TileOPs `origin/main` at `0d75579`
- CUDA: 13.2.1
- PyTorch: 2.13.0+cu132 (`torch.version.cuda == 13.2`)
- TileLang: 0.1.11+cu132.gitafcebed1
- cupti-python: 13.2.0
- cuda-bindings: 13.3.1
- GPU: NVIDIA H200

The official Dockerfile pins cupti-python 13.2.0. The local wrapper no longer
installs the previous 12.8.0 compatibility package.

## Persistent-container wrapper

`$HOST_HOME/foreman/local/tileops-container.sh` now uses the official
tag above and environment schema 3. All six schema-2 CUDA 12.9 containers were
removed after their host-mounted worktrees and `$CI_CACHE` were verified. Each
worktree will receive one new persistent schema-3 container; later commands use
`docker exec` in that container.

## Admission results

The official `scripts/ci/verify_runner_image.py` passed on an H200 and reported:

- torch 2.13.0+cu132 / CUDA 13.2;
- TileLang 0.1.11+cu132.gitafcebed1;
- FlashInfer 0.6.16.post3;
- mamba_ssm 2.3.2.post1;
- DeepGEMM 2.1.1;
- cuBLAS matmul, BMM, and einsum GPU smoke passed.

The five-round external baseline matrix also passed on GPU:

- vLLM 0.27.1 Triton `fused_experts`;
- vLLM Marlin W4A16;
- FLA 0.5.2 `chunk_gated_delta_rule`;
- mamba_ssm 2.3.2.post1 `mamba_chunk_scan_combined`;
- torch cuFFT;
- TileLang FFT compilation and correctness against cuFFT.

A native CUPTI benchmark of a 1024x1024 FP16 GEMM returned 200 samples with
median 0.007104 ms and metadata `timing=cupti`; CUDA-event fallback was not
used.

## Retest policy

The five original round agents retain ownership of their worktrees and PRs.
They are restarted sequentially. Each must rerun correctness plus the same
candidate/incumbent/external benchmark contract under this image, append raw
artifacts and an old-versus-new comparison to its round report, and fix its PR
only if the new official stack exposes a real compatibility or correctness
problem. GPU performance runs must not overlap across rounds.
