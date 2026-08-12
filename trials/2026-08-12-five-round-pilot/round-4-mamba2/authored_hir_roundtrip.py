from __future__ import annotations

from tilefoundry.module import module
from tilefoundry import func
from tilefoundry.target import CudaTarget
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401
from tilefoundry.ir.types.shard import (
    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,
)

@module(entry="mamba2_step", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132), Topology("thread", 512),))
class Mamba2StepRoundTrip:
    @func
    def mamba2_step(
        state: Tensor[(1, 2, 16, 16), "f32"],
        x_t: Tensor[(1, 2, 16), "f16"],
        dt_t: Tensor[(1, 2), "f32"],
        A: Tensor[(2,), "f32"],
        B_t: Tensor[(1, 1, 16), "f16"],
        C_t: Tensor[(1, 1, 16), "f16"]
    ):
        delta = softplus(dt_t)
        delta_2 = clamp(delta, min_val=0.0, max_val=inf)
        decay = reshape(A, new_shape=(1, 2))
        decay_2 = mul(delta_2, decay)
        decay_3 = exp(decay_2)
        next_state = reshape(decay_3, new_shape=(1, 2, 1, 1))
        next_state_2 = mul(next_state, state)
        update = reshape(delta_2, new_shape=(1, 2, 1, 1))
        update_2 = cast(x_t, dtype="f32")
        update_3 = reshape(update_2, new_shape=(1, 2, 16, 1))
        update_4 = mul(update, update_3)
        B_heads = repeat_interleave(B_t, repeats=2, axis=1)
        update_5 = cast(B_heads, dtype="f32")
        update_6 = reshape(update_5, new_shape=(1, 2, 1, 16))
        update_7 = mul(update_4, update_6)
        next_state_3 = add(next_state_2, update_7)
        C_heads = repeat_interleave(C_t, repeats=2, axis=1)
        y_t = cast(C_heads, dtype="f32")
        y_t_2 = reshape(y_t, new_shape=(1, 2, 1, 16))
        y_t_3 = mul(next_state_3, y_t_2)
        y_t_4 = reduce(y_t_3, axes=(-1,), keepdim=False, kind="sum")
        return (y_t_4, next_state_3)
