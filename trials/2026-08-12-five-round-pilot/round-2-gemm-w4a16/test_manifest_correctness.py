import pytest
import torch

from tileops.ops import GemmW4A16Op
from workloads.gemm import GemmW4A16Workload


CASES = [
    ("compile-smoke-square-64x64x128", 64, 64, 128),
    ("compile-smoke-rect-128x256x256", 128, 256, 256),
    ("decode-l2-resident-ish", 1, 8192, 8192),
    ("decode-hbm-streaming-threshold", 1, 8192, 16384),
    ("decode-non-power2-low-cta", 1, 7168, 20480),
    ("decode-long-k-pressure", 1, 8192, 81920),
]


@pytest.mark.parametrize("label,m,n,k", CASES, ids=[case[0] for case in CASES])
def test_manifest_row(label, m, n, k):
    del label
    torch.manual_seed(20260812 + k)
    workload = GemmW4A16Workload(m, n, k, torch.float16)
    inputs = workload.gen_inputs()
    op = GemmW4A16Op()
    actual = op(*inputs)
    expected = workload.ref_program(*inputs)
    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
    assert (op.kernel.decode_kernel is not None) == (m == 1)
    if m == 1:
        assert type(op.kernel.decode_kernel).__module__ == "tileops.kernels.gemm_w4a16_decode"
