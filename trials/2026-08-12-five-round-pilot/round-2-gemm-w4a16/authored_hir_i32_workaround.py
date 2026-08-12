"""TileFoundry-evaluable W4A16 HIR using i32 carriers for missing u8."""

import torch

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CudaTarget


M = 64
N = 64
K = 128


@module(
    entry="gemm_w4a16",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 128),),
)
class GemmW4A16:
    @func
    def gemm_w4a16(
        activation: Tensor[(M, K), "f16"],
        packed_weight: Tensor[(N, K // 2), "i32"],
        weight_scale: Tensor[(N, K // 128), "f32"],
        weight_zero: Tensor[(N, K // 128), "i32"],
    ) -> Tensor[(M, N), "f16"]:
        packed_i32 = tf.cast(packed_weight, dtype="i32")
        sixteen = tf.full_like(packed_i32, value=16)
        low = tf.mod(packed_i32, sixteen)
        high = tf.floor_div(packed_i32, sixteen)
        q_i32 = tf.reshape(tf.stack(low, high, axis=-1), new_shape=(N, K))
        scale_expanded = tf.repeat_interleave(weight_scale, repeats=128, axis=1)
        zero_expanded = tf.repeat_interleave(
            tf.cast(weight_zero, dtype="f32"), repeats=128, axis=1
        )
        dequant_f32 = (tf.cast(q_i32, dtype="f32") - zero_expanded) * scale_expanded
        dequant_f16 = tf.cast(dequant_f32, dtype="f16")
        return tf.matmul(activation, tf.transpose(dequant_f16, perm=(1, 0)))


@runtime_module(GemmW4A16)
class EvaluatorTwin:
    """Materializing twin used only to exercise TileFoundry check."""

    @runtime_func
    def gemm_w4a16(self, activation, packed_weight, weight_scale, weight_zero):
        packed_i32 = packed_weight.to(torch.int32)
        q_i32 = torch.stack((packed_i32 % 16, packed_i32 // 16), dim=-1).reshape(N, K)
        scale_expanded = weight_scale.repeat_interleave(128, dim=1)
        zero_expanded = weight_zero.float().repeat_interleave(128, dim=1)
        dequant_f16 = ((q_i32.float() - zero_expanded) * scale_expanded).half()
        return activation @ dequant_f16.T
