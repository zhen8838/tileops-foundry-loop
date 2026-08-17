from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget


CTAS = 132
N = CTAS * 128


# One mesh names one topology level. A mesh naming two -- ("cta", "thread") --
# is refused by `local_type_of`, so a kernel whose lanes each own part of a value
# states its cta placement here and leaves the lane structure to the kernel
# source. That gap is a finding, not something to work around silently.
@module(
    entry="kernel",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS),),
)
class OperatorModule:
    @func
    def kernel(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        # Author inside the mesh: a timeline has nothing to say about an op whose
        # result carries no position on the level being analysed.
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as cta:
            placed = tf.reshard(x, (N @ cta.block,), "gmem")
            # The gate wants the gmem-to-local movement stated. Note that
            # `analyze --timeline` and `schedule --topology` then refuse the
            # program, because the target publishes no rmem bandwidth; record
            # that as a finding and carry compute-cost, memory and roofline.
            local = tf.reshard(placed, (N @ cta.block,), "rmem")
            result = tf.square(local)
            return tf.reshard(result, (N @ cta.block,), "gmem")
