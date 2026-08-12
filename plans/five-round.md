# Goal: Five Sequential H200 Kernel Rounds

Read and follow `PLAYBOOK.md`. This plan supplies only the concrete operator
set and same-contract external baselines. Do not use or substitute a GQA
operator.

Pin the official image from `config/defaults.env` across the goal. Dispatch
each listed round to a Foreman solo agent using `gpt-5.6-sol` at `high` effort,
in this order:

| Round | Scope | TileOPs operator | Strongest same-contract baseline |
| ---: | --- | --- | --- |
| 1 | `MoE` | `FusedMoEExpertsNopadPersistent3WGFwdOp` | vLLM Triton `fused_experts` under the matching BF16/FP16 contract |
| 2 | `GEMM` | `GemmW4A16Op` | vLLM Marlin W4A16 for `M=1` |
| 3 | `LinearAttention` | `GatedDeltaNetPrefillFwdOp` | FLA `chunk_gated_delta_rule` on production BTHD workloads |
| 4 | `Mamba` | `Mamba2FwdOp` | `mamba_ssm` `mamba_chunk_scan_combined` |
| 5 | `FFT` | `FFTC2COp` | `torch.fft.fft` backed by cuFFT |

Use the Playbook's round completion, gap repair, and final reporting criteria
without modification.
