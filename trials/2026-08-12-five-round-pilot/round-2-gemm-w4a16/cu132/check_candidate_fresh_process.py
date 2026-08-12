import torch

from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel
from workloads.gemm import GemmW4A16Workload


m, n, k = 1, 8192, 81920
torch.manual_seed(20260812)
workload = GemmW4A16Workload(m, n, k, torch.float16)
inputs = workload.gen_inputs()
expected = workload.ref_program(*inputs)
actual = GemmW4A16Kernel(m, n, k, torch.float16)(*inputs)
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
diff = (actual.float() - expected.float()).abs()
print(
    "pass",
    "max_abs",
    diff.max().item(),
    "index7846",
    actual.flatten()[7846].item(),
    "expected7846",
    expected.flatten()[7846].item(),
)
