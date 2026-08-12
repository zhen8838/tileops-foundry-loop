import torch

from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel
from workloads.gemm import GemmW4A16Workload


m, n, k = 1, 8192, 81920
torch.manual_seed(20260812)
workload = GemmW4A16Workload(m, n, k, torch.float16)
inputs = workload.gen_inputs()
expected_runs = [workload.ref_program(*inputs) for _ in range(5)]
torch.cuda.synchronize()
print(
    "oracle_pairwise_max_abs",
    [
        (expected_runs[i].float() - expected_runs[0].float()).abs().max().item()
        for i in range(1, len(expected_runs))
    ],
)
expected = expected_runs[0]
kernel = GemmW4A16Kernel(m, n, k, torch.float16)
outputs = []
for run in range(50):
    actual = kernel(*inputs)
    torch.cuda.synchronize()
    outputs.append(actual.clone())
    diff = (actual.float() - expected.float()).abs()
    allowed = 0.07 + 0.05 * expected.float().abs()
    print(
        run,
        "failures",
        (diff > allowed).sum().item(),
        "max_abs",
        diff.max().item(),
        "index7846",
        actual.flatten()[7846].item(),
        "expected7846",
        expected.flatten()[7846].item(),
    )
print(
    "candidate_pairwise_max_abs",
    [(outputs[i].float() - outputs[0].float()).abs().max().item() for i in range(1, 50)],
)
