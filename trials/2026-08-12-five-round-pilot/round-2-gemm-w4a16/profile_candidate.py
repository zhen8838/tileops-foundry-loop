import json

import torch

from tileops.kernels.gemm_w4a16 import GemmW4A16Kernel
from workloads.gemm import GemmW4A16Workload


for label, n, k in (
    ("decode-l2-resident-ish", 8192, 8192),
    ("decode-long-k-pressure", 8192, 81920),
):
    torch.manual_seed(20260812 + k)
    workload = GemmW4A16Workload(1, n, k, torch.float16)
    inputs = workload.gen_inputs()
    candidate = GemmW4A16Kernel(1, n, k, torch.float16)
    candidate(*inputs)
    torch.cuda.synchronize()
    compiled = candidate.decode_kernel.kernel(**candidate.config)
    print(
        json.dumps(
            {
                "label": label,
                "config": candidate.config,
                "resource_usage": str(compiled.resource_usage),
            }
        ),
        flush=True,
    )

    trace_path = f"$CONTAINER_WORKSPACE/tileops/.artifacts/round-2-gemm-w4a16/{label}-trace.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(6):
            candidate(*inputs)
            prof.step()
    prof.export_chrome_trace(trace_path)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), flush=True)
    print(f"trace={trace_path}", flush=True)
    del candidate, inputs, workload
    torch.cuda.empty_cache()
