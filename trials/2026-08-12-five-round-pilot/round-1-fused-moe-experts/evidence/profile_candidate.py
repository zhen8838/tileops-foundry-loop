import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tileops.ops.moe import FusedMoEExpertsNopadPersistent3WGFwdOp
from workloads.moe import MoeExpertsWorkload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--incumbent", action="store_true")
    args = parser.parse_args()

    workload = MoeExpertsWorkload(
        args.tokens, args.experts, 8, 7168, 2048, torch.bfloat16
    )
    inputs = workload.gen_inputs()
    output = torch.empty(
        args.tokens, 7168, dtype=torch.bfloat16, device="cuda"
    )
    workspace = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    op = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens=args.tokens,
        num_experts=args.experts,
        top_k=8,
        hidden_size=7168,
        ffn_size=2048,
        use_fused_activation=not args.incumbent,
    )

    def run():
        op.forward(
            output,
            *inputs,
            expert_map=None,
            workspace1=workspace,
            workspace2=workspace,
            num_experts=args.experts,
        )

    run()
    run()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("tileops_fused_moe_candidate"):
            run()
        torch.cuda.synchronize()

    prof.export_chrome_trace(args.trace)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=40
    ))


if __name__ == "__main__":
    main()
