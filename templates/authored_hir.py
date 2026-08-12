from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget


N = 132 * 128


@module(
    entry="kernel",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 128)),
)
class OperatorModule:
    @func
    def kernel(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("cta", "thread"), (132, 128), ("block", "lane")) as mesh:
            local = tf.reshard(x, (N @ (mesh.block, mesh.lane),), "rmem")
            result = tf.square(local)
            return tf.reshard(result, (N @ (mesh.block, mesh.lane),), "gmem")
