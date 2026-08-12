"""Minimal authored-HIR attempt at the exact complex128 tensor contract."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


@module(
    entry="identity",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132),),
)
class Complex128Identity:
    @func
    def identity(x: Tensor[(1, 4), "complex128"]) -> Tensor[(1, 4), "complex128"]:
        return x
