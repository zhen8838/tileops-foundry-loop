"""Capture profiler and generated-source evidence for Gated DeltaNet prefill."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
    _prefill_blocksolve_A_bthd_tl,
    _prefill_chunk_local_cumsum_bthd_tl,
)
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.linear_attention import GatedDeltaNetPrefillFwdWorkload


def dump_generated_sources(seq_len: int, heads: int, output_dir: Path) -> None:
    from tileops.kernels.gated_deltanet.gdn_prefill.fused_fwd import (
        tilelang_fused_chunk_gdr_fwd,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    common = (1, heads, seq_len, 64, 128)
    kernels = {
        "chunk_cumsum": _prefill_chunk_local_cumsum_bthd_tl(
            1, heads, seq_len, 64, "bfloat16"
        ),
        "blocksolve_a": _prefill_blocksolve_A_bthd_tl(
            *common, "bfloat16"
        ),
    }
    if seq_len == 4096 and heads == 16:
        kernels["fused_chunk_gdr_fwd"] = tilelang_fused_chunk_gdr_fwd(
            heads,
            heads,
            128,
            128,
            64,
            1.0,
            qkva_dtype=torch.bfloat16,
            g_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            h0_dtype=torch.float32,
            ht_dtype=torch.float32,
            h_dtype=torch.bfloat16,
            o_dtype=torch.bfloat16,
            seqlen_dtype=torch.int32,
            accum_dtype="float32",
            use_initial_state=False,
            store_final_state=True,
            store_h=False,
            store_o=True,
            is_varlen=True,
            is_cp=False,
            block_DV=32,
        )
    else:
        from tileops.kernels.gated_deltanet.gdn_prefill.cp_fwd import (
            tilelang_correct_h0,
            tilelang_get_warmup_chunks,
            tilelang_zero_initial_cp_h0,
        )
        from tileops.kernels.gated_deltanet.gdn_prefill.prepare_h import (
            tilelang_prepare_h,
        )
        from tileops.kernels.gated_deltanet.gdn_prefill.utils import (
            tilelang_prepare_chunk_offsets,
        )

        kernels.update(
            {
                "get_warmup_chunks": tilelang_get_warmup_chunks(
                    num_heads=heads,
                    chunk_size=64,
                    threshold=-10.0,
                    accum_dtype="float32",
                    g_dtype=torch.bfloat16,
                    mask_dtype=torch.bool,
                    seqlen_dtype=torch.int32,
                ),
                "prepare_chunk_offsets": tilelang_prepare_chunk_offsets(
                    chunk_size=64,
                    block_size=8,
                    dtype=torch.int32,
                ),
                "prepare_h": tilelang_prepare_h(
                    heads,
                    heads,
                    128,
                    128,
                    64,
                    qkva_dtype=torch.bfloat16,
                    g_dtype=torch.bfloat16,
                    b_dtype=torch.bfloat16,
                    h0_dtype=torch.float32,
                    ht_dtype=torch.bfloat16,
                    h_dtype=torch.bfloat16,
                    seqlen_dtype=torch.int32,
                    accum_dtype="float32",
                    use_initial_state=False,
                    store_final_state=True,
                    store_h=False,
                    is_varlen=True,
                    is_cp=True,
                ),
                "correct_h0": tilelang_correct_h0(
                    H=heads,
                    DK=128,
                    DV=128,
                    res_dtype=torch.float32,
                    accum_dtype="float32",
                    buffer_dtype=torch.bfloat16,
                    seqlen_dtype=torch.int32,
                    mask_dtype=torch.bool,
                    use_raw_h0=False,
                ),
                "zero_initial_cp_h0": tilelang_zero_initial_cp_h0(
                    H=heads,
                    DK=128,
                    DV=128,
                    res_dtype=torch.float32,
                    seqlen_dtype=torch.int32,
                ),
                "fused_chunk_gdr_fwd": tilelang_fused_chunk_gdr_fwd(
                    heads,
                    heads,
                    128,
                    128,
                    64,
                    1.0,
                    qkva_dtype=torch.bfloat16,
                    g_dtype=torch.bfloat16,
                    b_dtype=torch.bfloat16,
                    h0_dtype=torch.float32,
                    ht_dtype=torch.float32,
                    h_dtype=torch.bfloat16,
                    o_dtype=torch.bfloat16,
                    seqlen_dtype=torch.int32,
                    accum_dtype="float32",
                    use_initial_state=True,
                    store_final_state=True,
                    store_h=False,
                    store_o=True,
                    is_varlen=True,
                    is_cp=True,
                    block_DV=128,
                ),
            }
        )

    resource_usage = {}
    for name, kernel in kernels.items():
        (output_dir / f"{name}.cu").write_text(kernel.get_kernel_source())
        usage = kernel.resource_usage
        resource_usage[name] = str(usage() if callable(usage) else usage)
    (output_dir / "resource-usage.json").write_text(
        json.dumps(resource_usage, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TILEOPS_GDN_PREFILL_FORCE_PARTITION"] = "0"
    torch.manual_seed(42)
    workload = GatedDeltaNetPrefillFwdWorkload(
        1,
        args.heads,
        args.seq_len,
        128,
        128,
        64,
        torch.bfloat16,
        layout="bthd",
    )
    inputs = workload.gen_inputs()
    op = GatedDeltaNetPrefillFwdOp(chunk_size=64, layout="bthd")

    for _ in range(3):
        outputs = op(*inputs)
    torch.cuda.synchronize()
    del outputs

    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        outputs = op(*inputs)
        torch.cuda.synchronize()

    stem = f"s{args.seq_len}-h{args.heads}-bf16"
    prof.export_chrome_trace(str(args.output_dir / f"{stem}-trace.json"))
    cuda_events = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    kernels = []
    for event in cuda_events:
        kernels.append(
            {
                "name": event.name,
                "duration_us": event.device_time_total,
            }
        )
    summary = {
        "shape": {
            "batch": 1,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "dim_k": 128,
            "dim_v": 128,
            "chunk_size": 64,
            "dtype": "bfloat16",
        },
        "launch_count": len(cuda_events),
        "peak_allocated_delta_bytes": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ),
        "output_shapes": [list(tensor.shape) for tensor in outputs],
        "kernels": kernels,
    }
    (args.output_dir / f"{stem}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    dump_generated_sources(args.seq_len, args.heads, args.output_dir / "generated")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
