import torch

from benchmarks.ops.bench_gemm import _prepare_marlin_w4a16_baseline
from workloads.gemm import GemmW4A16Workload


for shape in ((1, 64, 128), (1, 8192, 8192)):
    m, n, k = shape
    torch.manual_seed(20260812)
    workload = GemmW4A16Workload(m, n, k, torch.float16)
    inputs = workload.gen_inputs()
    expected = workload.ref_program(*inputs)
    for name, fp32 in (("fp16", False), ("fp32", True)):
        fn, args = _prepare_marlin_w4a16_baseline(m, n, k, fp32, *inputs)
        actual = fn(*args)
        torch.cuda.synchronize()
        diff = (actual.float() - expected.float()).abs()
        rel = diff / expected.float().abs().clamp_min(1e-12)
        torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
        print(shape, name, "max_abs", diff.max().item(), "max_rel", rel.max().item())
