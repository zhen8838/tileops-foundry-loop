# First candidate: direct runtime-indexed routed experts

Saved: `2026-08-12T00:37:16+08:00`

## Blind-phase record

The intended blind phase was compromised before this file was saved. At
`2026-08-12T00:09:37+08:00`, an `rg` command intended to select unrelated
kernel examples used repository-relative negative globs that did not exclude
`src/tileops/kernels/grouped_gemm/grouped_gemm.py`. Its output contained only
file paths and line numbers for `@T.prim_func`, `class`, and `def` matches; it
did not print an incumbent kernel body or configuration. The command still
searched a grouped-GEMM file, which violates the brief's literal blind-phase
rule. No incumbent idea was adopted in this candidate. Incumbent bodies were
not opened before the evidence below was recorded.

## Authored HIR

- H200 authored graph: `authored_hir.py:RoutedExperts.routed_experts`
- Evaluator twin: `authored_hir_twin.py:RoutedExpertsTwin.routed_experts`
- Check inputs/checkpoint: `hidden.pt`, `topk_weights.pt`, `topk_ids.pt`,
  `model.safetensors`

`tilefoundry analyze authored_hir.py:RoutedExperts --compute-cost --memory
--roofline --json` passed type inference and reported the H200 target, 99,072
BF16 FLOPs, 1,792 FP32 FLOPs, 209,504 bytes read, 110,080 bytes written, and a
67 ns memory roofline for the deliberately small evaluator shape.

Host-side H200 `check` could not create a cuBLAS handle while the persistent
benchmark container owned the GPU. The identical immutable Module therefore
has a `CpuTarget` alias used only by the evaluator. This command passed:

```text
tilefoundry check authored_hir_twin.py:RoutedExpertsTwin.routed_experts \
  --input hidden.pt --input topk_weights.pt --input topk_ids.pt --ckpt . \
  --out output --fn allclose --atol 0.01 --rtol 0.01 --fn nan_inf --json

reference: evaluator on RoutedExperts.routed_experts
output: bf16[4,64]
allclose: max_violation=0.0, passed=true
nan_inf: nan=0, inf=0, passed=true
overall: PASS
```

## TileLang runtime twin

New file:

- `first_candidate_direct.py` (preserved in this shared round directory; the
  rejected 953 ms candidate was removed from the production worktree)

`FusedMoeExpertsDirectKernel` has two launches. The first follows runtime
`topk_ids` to compute gate/up and SiLU into a BF16 `[T*K,F]` temporary. The
second performs down projection, route weighting, FP32 route reduction, routed
scaling, and the final BF16 store to the caller's output buffer.

Small-shape correctness used BF16, repeated expert ids, and scaling 0.75:

```text
FusedMoeExpertsDirectKernel initialized with config: {'threads': 128}
max_abs 1.1920928955078125e-07
allclose True
finite True
```

The first compile failed because postponed annotations hid closure dimensions
from TileLang's eager `get_type_hints`; removing the future annotation fixed
the builder without changing the kernel. Both successful builds emit a
conservative TileLang data-race warning for predicated fragment stores. The
numerical result above shows no observed race; this remains a compiler
ergonomics gap to record.

## Raw latency

One production row was measured before inspecting or running the incumbent:

```text
workload: qwen3-235b-decode
shape: T=512 E=128 K=8 H=7168 F=2048 dtype=bf16
compile_plus_warmup_s 8.322015929035842
samples_ms [953.0823364257812, 953.0839233398438, 953.0324096679688]
median_ms 953.0823364257812
finite True
```

Timing used CUDA events only for this early rejection measurement. It is not
used in the final candidate/incumbent/vLLM comparison, which must use the
repository CUPTI harness in one process.

## Known limitations

- BF16, SILU, SM90, single-GPU only.
- The internal BF16 temporary is lazily allocated and cached by the candidate
  Kernel object; the public Op has not been changed and does not dispatch here.
- No expert grouping: each route rereads its selected expert weights. The
  953.08 ms decode result demonstrates that this is not a viable production
  algorithm.
- TileFoundry evaluates general vector `IndexSelect`, but current HIR-to-TIR
  lowering supports only a one-element index and unit leading dimensions. The
  handwritten runtime twin is therefore required for the vector of `T*K`
  runtime routes.
- FP16, GELU, `expert_map`, public fallback behavior, and production-size
  oracle correctness are not claims of this first candidate.
