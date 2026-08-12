# Round 2 first correct candidate

Saved while the incumbent implementation remained unread. This file is immutable
after the blind phase ends.

## Authored HIR and TileFoundry validation

- Exact-contract authored HIR:
  `$TRIAL_SOURCE/round-2-gemm-w4a16/authored_hir.py`
- Static i32-carrier workaround used to exercise the current checker:
  `$TRIAL_SOURCE/round-2-gemm-w4a16/authored_hir_i32_workaround.py`
- The graph preserves little-nibble-first stack/reshape, group-128 metadata
  expansion, FP32 affine dequantization, the FP16 rounding point, transpose, and
  matmul.
- Exact HIR import fails before parsing the body because current TileFoundry has no
  `u8` dtype. Raw log: `hir-import.log`.
- Dynamic i32 workaround then fails because `RepeatInterleave` shape inference
  attempts Python `Call * int` for the dynamic `K//128` axis. Raw logs:
  `hir-i32-analyze-smoke.json` (first failed version) and
  `hir-i32-evaluator-smoke.log` (first failed version, later overwritten by the
  passing static run; traceback retained in the execution transcript).
- Static `(64,64,128)` i32 workaround passes evaluator exactly (`max_abs=0`) and
  `tilefoundry check` exactly (`atol=0`, `rtol=0`, no NaN/Inf). Raw outputs:
  `hir-i32-evaluator-smoke.log`, `hir-i32-check-smoke-2.json`, and
  `hir-i32-analyze-smoke.json`.
- `schedule` reaches target costing but fails because the H200 target declares no
  dense i32 peak rate. Raw output: `hir-i32-schedule-smoke-3.json`.

## Runtime twin

- Handwritten fused TileLang twin:
  `$HOST_HOME/TileOPs-worktrees/tileops-tileops-r2-gemm-w4a16/src/tileops/kernels/gemm_w4a16_r2.py`
- Public dispatch changed only to bind `GemmW4A16Op` to that new class; the
  candidate does not import or call the incumbent.
- Generated smoke CUDA source:
  `$TRIAL_SOURCE/round-2-gemm-w4a16/first-candidate-generated-smoke.cu`
- The timed kernel is one launch and holds only a 128-element activation shared
  tile plus per-CTA fragments. It never allocates or emits an `[N,K]` output.
- Initial configuration: `block_n=8`, `threads=128`; tuning count before this
  snapshot: 0 configurations beyond the initial choice.

## Correctness

Compile-smoke `(M,N,K)=(64,64,128)`:

```text
1 passed, 37 deselected in 7.97s
```

JUnit: `first-smoke.xml`. Raw output: `first-smoke-3.log`.

`decode-l2-resident-ish (1,8192,8192)` on NVIDIA H200:

```text
candidate_module=tileops.kernels.gemm_w4a16_r2
candidate_class=GemmW4A16Kernel
candidate_config={'block_n': 8, 'threads': 128}
shape=(1, 8192) dtype=torch.float16
max_abs=0.03125
max_rel=0.00508044008
allclose=True atol=0.07 rtol=0.05
```

Raw correctness and timing output: `first-candidate-decode-l2.log`.

## Repeated raw latency

Repository-native CUPTI attribution, L2 reset and shifted input pointers before
every sample, compilation excluded:

| run | samples | median ms | p10 ms | p90 ms | timer |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 145 | 0.064705 | 0.063937 | 0.065344 | cupti |
| 2 | 137 | 0.064800 | 0.064064 | 0.065408 | cupti |
| 3 | 157 | 0.064705 | 0.064096 | 0.065312 | cupti |

Every individual sample is preserved in `first-candidate-decode-l2.log`.
