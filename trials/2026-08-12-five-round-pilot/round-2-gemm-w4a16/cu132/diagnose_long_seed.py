import torch

from benchmarks.ops.bench_gemm import _prepare_marlin_w4a16_baseline
from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel, _gemm_w4a16_kernel
from workloads.gemm import GemmW4A16Workload


m, n, k = 1, 8192, 81920
torch.manual_seed(20260812)
workload = GemmW4A16Workload(m, n, k, torch.float16)
inputs = workload.gen_inputs()
expected = workload.ref_program(*inputs)
config = {
    "block_m": 64,
    "block_n": 64,
    "block_k": 64,
    "num_stages": 2,
    "threads": 128,
}
functions = {
    "candidate": (GemmW4A16Kernel(m, n, k, torch.float16), inputs),
    "base-incumbent": (_gemm_w4a16_kernel(m, n, k, "float16")(**config), inputs),
    "marlin-fp16": _prepare_marlin_w4a16_baseline(m, n, k, False, *inputs),
    "marlin-fp32": _prepare_marlin_w4a16_baseline(m, n, k, True, *inputs),
}
candidate_index = None
for name, (fn, args) in functions.items():
    actual = fn(*args)
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    allowed = 0.07 + 0.05 * expected.float().abs()
    failures = diff > allowed
    worst = (diff - allowed).argmax()
    if name == "candidate":
        candidate_index = diff.argmax()
    print(
        name,
        "failures",
        failures.sum().item(),
        "max_abs",
        diff.max().item(),
        "max_excess",
        (diff - allowed).max().item(),
        "worst_index",
        worst.item(),
        "actual",
        actual.flatten()[worst].item(),
        "expected",
        expected.flatten()[worst].item(),
        "allowed",
        allowed.flatten()[worst].item(),
    )
    if candidate_index is not None:
        print(
            name,
            "at_candidate_max_abs_index",
            candidate_index.item(),
            "actual",
            actual.flatten()[candidate_index].item(),
            "expected",
            expected.flatten()[candidate_index].item(),
            "abs_diff",
            diff.flatten()[candidate_index].item(),
        )
