"""Authored HIR for the Gated DeltaNet prefill recurrence.

The executable boundary is one ordered step.  The carry and all arithmetic are
f32; only q/k/v/g/beta use the model dtype.  ``ordered_scan_repro.py`` records
the corresponding multi-step contract and the current parser result.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bindings used by @func
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


B = 1
H = 2
DK = 16
DV = 16
DT = "f16"


@module(
    entry="delta_step",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 512)),
)
class GatedDeltaNetStep:
    @func
    def delta_step(
        state: Tensor[(B, H, DK, DV), "f32"],
        q_t: Tensor[(B, H, DK), DT],
        k_t: Tensor[(B, H, DK), DT],
        v_t: Tensor[(B, H, DV), DT],
        g_t: Tensor[(B, H), DT],
        beta_t: Tensor[(B, H), DT],
    ):
        alpha = tf.exp(tf.cast(g_t, dtype="f32"))
        qf = tf.cast(q_t, dtype="f32")
        kf = tf.cast(k_t, dtype="f32")
        vf = tf.cast(v_t, dtype="f32")
        betaf = tf.cast(beta_t, dtype="f32")
        old = tf.reshape(
            tf.matmul(tf.reshape(kf, new_shape=(B, H, 1, DK)), state),
            new_shape=(B, H, DV),
        )
        delta = tf.reshape(betaf, new_shape=(B, H, 1)) * (
            vf - tf.reshape(alpha, new_shape=(B, H, 1)) * old
        )
        next_state = (
            tf.reshape(alpha, new_shape=(B, H, 1, 1)) * state
            + tf.reshape(kf, new_shape=(B, H, DK, 1))
            * tf.reshape(delta, new_shape=(B, H, 1, DV))
        )
        out = tf.reshape(
            tf.matmul(tf.reshape(qf, new_shape=(B, H, 1, DK)), next_state),
            new_shape=(B, H, DV),
        )
        return out, next_state
