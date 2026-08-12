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

@module(entry="dft_pair_f32", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132), Topology("thread", 256),))
class FFTPairF32:
    @func
    def complex_butterfly(
        even_r: Tensor[(2, 2), "f32"],
        even_i: Tensor[(2, 2), "f32"],
        odd_r: Tensor[(2, 2), "f32"],
        odd_i: Tensor[(2, 2), "f32"],
        tw_r: Tensor[(2,), "f32"],
        tw_i: Tensor[(2,), "f32"]
    ):
        prod_r = mul(odd_r, tw_r)
        prod_r_2 = mul(odd_i, tw_i)
        prod_r_3 = sub(prod_r, prod_r_2)
        out_r = add(even_r, prod_r_3)
        out_r_2 = sub(even_r, prod_r_3)
        out_r_3 = concat(out_r, out_r_2, axis=1)
        prod_i = mul(odd_r, tw_i)
        prod_i_2 = mul(odd_i, tw_r)
        prod_i_3 = add(prod_i, prod_i_2)
        out_i = add(even_i, prod_i_3)
        out_i_2 = sub(even_i, prod_i_3)
        out_i_3 = concat(out_i, out_i_2, axis=1)
        return (out_r_3, out_i_3)

    @func
    def dft_pair_f32(
        x_r: Tensor[(2, 4), "f32"],
        x_i: Tensor[(2, 4), "f32"],
        w_r: Tensor[(4, 4), "f32"],
        w_i: Tensor[(4, 4), "f32"]
    ):
        y_r = matmul(x_r, w_r)
        y_r_2 = matmul(x_i, w_i)
        y_r_3 = sub(y_r, y_r_2)
        y_i = matmul(x_r, w_i)
        y_i_2 = matmul(x_i, w_r)
        y_i_3 = add(y_i, y_i_2)
        return (y_r_3, y_i_3)
