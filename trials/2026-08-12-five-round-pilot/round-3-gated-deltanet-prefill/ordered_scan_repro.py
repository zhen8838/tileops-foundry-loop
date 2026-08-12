"""Smallest authored-HIR attempt at an ordered Gated DeltaNet carry scan."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bindings used by @func


B = 1
S = 4
H = 2
DK = 16
DV = 16
DT = "f16"


@module(entry="gated_deltanet_prefill")
class GatedDeltaNetPrefill:
    @func
    def gated_deltanet_prefill(
        q: Tensor[(B, S, H, DK), DT],
        k: Tensor[(B, S, H, DK), DT],
        v: Tensor[(B, S, H, DV), DT],
        g: Tensor[(B, S, H), DT],
        beta: Tensor[(B, S, H), DT],
    ):
        state = tf.zeros(shape=(B, H, DK, DV), dtype="f32")
        out = tf.zeros(shape=(B, S, H, DV), dtype="f32")
        for t in tile(S, 1):
            alpha = tf.exp(tf.cast(g[:, t, :], dtype="f32"))
            qf = tf.cast(q[:, t, :, :], dtype="f32")
            kf = tf.cast(k[:, t, :, :], dtype="f32")
            vf = tf.cast(v[:, t, :, :], dtype="f32")
            old = tf.reshape(
                tf.matmul(tf.reshape(kf, new_shape=(B, H, 1, DK)), state),
                new_shape=(B, H, DV),
            )
            delta = tf.reshape(tf.cast(beta[:, t, :], dtype="f32"), new_shape=(B, H, 1)) * (
                vf - tf.reshape(alpha, new_shape=(B, H, 1)) * old
            )
            state = (
                tf.reshape(alpha, new_shape=(B, H, 1, 1)) * state
                + tf.reshape(kf, new_shape=(B, H, DK, 1))
                * tf.reshape(delta, new_shape=(B, H, 1, DV))
            )
            out = tf.insert_slice(
                out,
                tf.reshape(
                    tf.matmul(tf.reshape(qf, new_shape=(B, H, 1, DK)), state),
                    new_shape=(B, 1, H, DV),
                ),
                offsets=(0, t, 0, 0),
            )
        return tf.cast(out, dtype=DT), tf.cast(state, dtype=DT)
