from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.target import CudaTarget


M, N, K = 1, 4096, 4096


@module(entry="kernel", target=CudaTarget("nvidia.h200_sxm"))
class OperatorModule:
    @func
    def kernel(
        x: Tensor[(M, K), "bf16"],
        weight: ConstTensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        return tf.matmul(x, weight)
