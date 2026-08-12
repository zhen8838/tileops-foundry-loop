"""Small typed HIR for the routed-expert value graph used by the example."""

from dataclasses import replace

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, Topology, tf
from tilefoundry.target import CpuTarget, CudaTarget


T, E, K, H, F = 4, 4, 2, 64, 32
DT = "bf16"
ROUTED_SCALING_FACTOR = 0.75


@module(
    entry="routed_experts",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 256)),
)
class RoutedExperts:
    @func
    def routed_experts(
        hidden: Tensor[(T, H), DT],
        topk_weights: Tensor[(T, K), "f32"],
        topk_ids: Tensor[(T, K), "i32"],
        w_gate_up: ConstTensor[(E, 2 * F, H), DT],
        w_down: ConstTensor[(E, H, F), DT],
    ) -> Tensor[(T, H), DT]:
        flat_ids = tf.reshape(topk_ids, new_shape=(T * K,))
        selected_in = tf.reshape(
            tf.index_select(w_gate_up, flat_ids, dim=0),
            new_shape=(T, K, 2 * F, H),
        )
        both = tf.reshape(
            tf.matmul(selected_in, tf.reshape(hidden, new_shape=(T, 1, H, 1))),
            new_shape=(T, K, 2 * F),
        )
        inner = tf.silu(both[:, :, :F]) * both[:, :, F:]
        selected_down = tf.reshape(
            tf.index_select(w_down, flat_ids, dim=0),
            new_shape=(T, K, H, F),
        )
        down = tf.reshape(
            tf.matmul(selected_down, tf.reshape(inner, new_shape=(T, K, F, 1))),
            new_shape=(T, K, H),
        )
        weighted = tf.cast(down, dtype="f32") * tf.reshape(
            topk_weights, new_shape=(T, K, 1)
        )
        mixed = tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")
        scaled = mixed * tf.full_like(mixed, value=ROUTED_SCALING_FACTOR)
        return tf.cast(scaled, dtype=DT)


# Host-side checks avoid competing with benchmark agents for an H200.
RoutedExpertsCheck = replace(RoutedExperts, target=CpuTarget())
