"""Minimal authored-HIR attempt at an exact real-pair complex128 contract."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


@module(
    entry="pair_add",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132),),
)
class F64Pair:
    @func
    def pair_add(
        x_r: Tensor[(1, 4), "f64"],
        x_i: Tensor[(1, 4), "f64"],
    ):
        return tf.add(x_r, x_r), tf.add(x_i, x_i)
