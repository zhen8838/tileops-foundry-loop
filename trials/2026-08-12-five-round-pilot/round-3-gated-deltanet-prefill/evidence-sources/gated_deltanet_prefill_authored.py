"""TileFoundry-derived runtime twin for production Gated DeltaNet prefill.

The authored HIR is preserved in the round evidence directory.  This first
runtime twin maps one ordered recurrence stream to one CTA and keeps its FP32
state in shared memory.  It intentionally targets only the production BTHD
specialization; the public Op retains its general implementation for all other
calls.
"""

from __future__ import annotations

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["GatedDeltaNetPrefillAuthoredKernel"]


def _gated_deltanet_prefill_authored_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
):
    @tilelang.jit(out_idx=[5, 6])
    def _func(threads: int):
        @T.prim_func
        def main(
            q: T.Tensor[(batch, seq_len, heads, dim_k), dtype],
            k: T.Tensor[(batch, seq_len, heads, dim_k), dtype],
            v: T.Tensor[(batch, seq_len, heads, dim_v), dtype],
            g: T.Tensor[(batch, seq_len, heads), dtype],
            beta: T.Tensor[(batch, seq_len, heads), dtype],
            o: T.Tensor[(batch, seq_len, heads, dim_v), dtype],
            final_state: T.Tensor[(batch, heads, dim_k, dim_v), dtype],
        ):
            with T.Kernel(batch * heads, threads=threads) as stream:
                b = stream // heads
                h = stream % heads
                state = T.alloc_shared((dim_k, dim_v), "float32")
                q_shared = T.alloc_shared((dim_k,), dtype)
                k_shared = T.alloc_shared((dim_k,), dtype)
                v_shared = T.alloc_shared((dim_v,), dtype)
                delta_shared = T.alloc_shared((dim_v,), "float32")
                scalars = T.alloc_shared((2,), "float32")
                old = T.alloc_fragment((dim_v,), "float32")
                read = T.alloc_fragment((dim_v,), "float32")

                for i, j in T.Parallel(dim_k, dim_v):
                    state[i, j] = T.float32(0)
                T.sync_threads()

                for t in T.Serial(seq_len):
                    for i in T.Parallel(dim_k):
                        q_shared[i] = q[b, t, h, i]
                        k_shared[i] = k[b, t, h, i]
                    for j in T.Parallel(dim_v):
                        v_shared[j] = v[b, t, h, j]
                    for j in T.Parallel(2):
                        scalars[j] = T.if_then_else(
                            j == 0,
                            T.exp(T.cast(g[b, t, h], "float32")),
                            T.cast(beta[b, t, h], "float32"),
                        )
                    T.sync_threads()

                    for j in T.Parallel(dim_v):
                        old[j] = T.float32(0)
                        for i in T.Serial(dim_k):
                            old[j] = old[j] + T.cast(k_shared[i], "float32") * state[i, j]
                        delta_shared[j] = scalars[1] * (
                            T.cast(v_shared[j], "float32") - scalars[0] * old[j]
                        )
                    T.sync_threads()

                    for i, j in T.Parallel(dim_k, dim_v):
                        state[i, j] = (
                            scalars[0] * state[i, j]
                            + T.cast(k_shared[i], "float32") * delta_shared[j]
                        )
                    T.sync_threads()

                    for j in T.Parallel(dim_v):
                        read[j] = T.float32(0)
                        for i in T.Serial(dim_k):
                            read[j] = read[j] + T.cast(q_shared[i], "float32") * state[i, j]
                        o[b, t, h, j] = T.cast(read[j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(dim_k, dim_v):
                    final_state[b, h, i, j] = T.cast(state[i, j], dtype)

        return main

    return _func


class GatedDeltaNetPrefillAuthoredKernel(Kernel):
    """Ordered FP32-carry runtime twin for the production BTHD specialization."""

    supported_archs = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str,
        layout: str,
        tune: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if (batch, chunk_size, dim_k, dim_v, layout) != (1, 64, 128, 128, "bthd"):
            raise ValueError(
                "GatedDeltaNetPrefillAuthoredKernel serves only "
                "B=1, chunk_size=64, DK=DV=128, layout='bthd'"
            )
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"Unsupported production dtype: {dtype}")
        if tune:
            raise ValueError("The blind authored runtime has no tuning surface")
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype_name = dtype
        self.kernel = _gated_deltanet_prefill_authored_kernel(
            batch, heads, seq_len, dim_k, dim_v, dtype
        )
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(self.config["threads"])(q, k, v, g, beta)
