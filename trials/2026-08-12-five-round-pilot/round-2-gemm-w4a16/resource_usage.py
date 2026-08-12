import torch

from tileops.kernels.gemm_w4a16_decode import GemmW4A16DecodeKernel


for name, shape in (("l2", (1, 4096, 4096)), ("long", (1, 4096, 14336))):
    kernel = GemmW4A16DecodeKernel(*shape, torch.float16)
    compiled = kernel.kernel(**kernel.config)
    print(name, shape, kernel.config)
    print("resource_usage", compiled._primary_resource_usage())
    for line in compiled.get_host_source().splitlines():
        if "main_kernel" in line or "cudaFuncSetAttribute" in line or "<<<" in line:
            print(line.strip())
