"""Rejected direct runtime-indexed fused-MoE TileLang candidate.

This is the handwritten runtime twin of the round-1 TileFoundry authored HIR.
It deliberately follows the route value graph without first sorting routes by
expert: gate/up and SiLU are one kernel, then down projection, route weighting,
and reduction are one kernel.  The implementation is useful as a semantic and
lowering baseline; route grouping is required before it can be competitive at
production dimensions.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["FusedMoeExpertsDirectKernel"]


@functools.lru_cache(maxsize=32)
def _direct_gate_up_silu_kernel(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    ffn_size: int,
    dtype: str,
) -> Callable:
    num_routes = num_tokens * top_k

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(threads: int = 128) -> Callable:
        @T.prim_func
        def main(
            hidden: T.Tensor((num_tokens, hidden_size), dtype),  # type: ignore
            w_gate_up: T.Tensor((num_experts, 2 * ffn_size, hidden_size), dtype),  # type: ignore
            topk_ids: T.Tensor((num_tokens, top_k), "int32"),  # type: ignore
            inner: T.Tensor((num_routes, ffn_size), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(ffn_size, threads),
                num_routes,
                threads=threads,
            ) as (bx, route):
                gate = T.alloc_fragment((threads,), "float32")
                up = T.alloc_fragment((threads,), "float32")
                T.clear(gate)
                T.clear(up)
                token = route // top_k
                route_slot = route % top_k
                expert = topk_ids[token, route_slot]
                for lane in T.Parallel(threads):
                    f = bx * threads + lane
                    if f < ffn_size:
                        for h in T.serial(hidden_size):
                            x = T.cast(hidden[token, h], "float32")
                            gate[lane] += x * T.cast(
                                w_gate_up[expert, f, h],
                                "float32",
                            )
                            up[lane] += x * T.cast(
                                w_gate_up[expert, ffn_size + f, h],
                                "float32",
                            )
                        inner[route, f] = T.cast(
                            gate[lane] * T.sigmoid(gate[lane]) * up[lane],
                            dtype,
                        )

        return main

    return build


@functools.lru_cache(maxsize=32)
def _direct_down_mix_kernel(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    ffn_size: int,
    routed_scaling_factor: float,
    dtype: str,
) -> Callable:
    num_routes = num_tokens * top_k

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(threads: int = 128) -> Callable:
        @T.prim_func
        def main(
            inner: T.Tensor((num_routes, ffn_size), dtype),  # type: ignore
            w_down: T.Tensor((num_experts, hidden_size, ffn_size), dtype),  # type: ignore
            topk_weights: T.Tensor((num_tokens, top_k), "float32"),  # type: ignore
            topk_ids: T.Tensor((num_tokens, top_k), "int32"),  # type: ignore
            output: T.Tensor((num_tokens, hidden_size), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(hidden_size, threads),
                num_tokens,
                threads=threads,
            ) as (bx, token):
                mixed = T.alloc_fragment((threads,), "float32")
                route_down = T.alloc_fragment((threads,), "float32")
                T.clear(mixed)
                for route_slot in T.serial(top_k):
                    T.clear(route_down)
                    route = token * top_k + route_slot
                    expert = topk_ids[token, route_slot]
                    for lane in T.Parallel(threads):
                        h = bx * threads + lane
                        if h < hidden_size:
                            for f in T.serial(ffn_size):
                                route_down[lane] += T.cast(
                                    inner[route, f],
                                    "float32",
                                ) * T.cast(
                                    w_down[expert, h, f],
                                    "float32",
                                )
                            mixed[lane] += (
                                route_down[lane] * topk_weights[token, route_slot]
                            )
                for lane in T.Parallel(threads):
                    h = bx * threads + lane
                    if h < hidden_size:
                        output[token, h] = T.cast(
                            mixed[lane] * routed_scaling_factor,
                            dtype,
                        )

        return main

    return build


class FusedMoeExpertsDirectKernel(Kernel):
    """BF16/SILU/single-GPU runtime twin for fused routed experts."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype,
        routed_scaling_factor: float = 1.0,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if dtype is not torch.bfloat16:
            raise ValueError(
                "FusedMoeExpertsDirectKernel supports only torch.bfloat16, "
                f"got {dtype}"
            )
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.routed_scaling_factor = routed_scaling_factor
        self._inner: Optional[torch.Tensor] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    def forward(
        self,
        output: torch.Tensor,
        hidden: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        expected_inner = (self.num_tokens * self.top_k, self.ffn_size)
        if (
            self._inner is None
            or self._inner.device != hidden.device
            or self._inner.shape != expected_inner
        ):
            self._inner = torch.empty(
                expected_inner,
                dtype=self.dtype,
                device=hidden.device,
            )

        gate_up = _direct_gate_up_silu_kernel(
            self.num_tokens,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.ffn_size,
            self.dtype_str,
        )(self.config["threads"])
        down_mix = _direct_down_mix_kernel(
            self.num_tokens,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.ffn_size,
            self.routed_scaling_factor,
            self.dtype_str,
        )(self.config["threads"])
        gate_up(hidden, w_gate_up, topk_ids, self._inner)
        down_mix(self._inner, w_down, topk_weights, topk_ids, output)
