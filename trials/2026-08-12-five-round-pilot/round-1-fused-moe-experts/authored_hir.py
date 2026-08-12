"""Authored HIR reference for the TileOPs fused routed-expert contract.

The dimensions are deliberately small enough for evaluator checks.  The value
graph, dtypes, weight orientation, runtime int32 expert selection, FP32 weighted
reduction, and final routed scaling match the production manifest contract.
"""

from dataclasses import replace

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, Topology, tf
from tilefoundry.target import CpuTarget, CudaTarget


T = 4
E = 4
K = 2
H = 64
F = 32
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
        hidden_col = tf.reshape(hidden, new_shape=(T, 1, H, 1))
        both = tf.reshape(
            tf.matmul(selected_in, hidden_col),
            new_shape=(T, K, 2 * F),
        )
        gate = both[:, :, :F]
        up = both[:, :, F:]
        inner = tf.silu(gate) * up
        selected_down = tf.reshape(
            tf.index_select(w_down, flat_ids, dim=0),
            new_shape=(T, K, H, F),
        )
        down = tf.reshape(
            tf.matmul(
                selected_down,
                tf.reshape(inner, new_shape=(T, K, F, 1)),
            ),
            new_shape=(T, K, H),
        )
        weighted = tf.cast(down, dtype="f32") * tf.reshape(
            topk_weights,
            new_shape=(T, K, 1),
        )
        mixed = tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")
        scaled = mixed * tf.full_like(mixed, value=ROUTED_SCALING_FACTOR)
        return tf.cast(scaled, dtype=DT)


# ``check`` executes on the selected target.  The persistent benchmark container
# owns the GPU, so the host-side evaluator uses this identical CPU-target alias;
# H200 analysis continues to select ``RoutedExperts`` above.
RoutedExpertsCheck = replace(RoutedExperts, target=CpuTarget())
