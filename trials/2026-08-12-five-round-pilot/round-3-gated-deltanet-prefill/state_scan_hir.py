"""Executable HIR proof that ordered FP32 carry itself is supported."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bindings used by @func


B = 1
S = 4
H = 2
DK = 16
DV = 16
DT = "f16"


@module(entry="state_scan")
class StateScan:
    @func
    def state_scan(
        k: Tensor[(B, S, H, DK), DT],
        v: Tensor[(B, S, H, DV), DT],
        g: Tensor[(B, S, H), DT],
        beta: Tensor[(B, S, H), DT],
    ):
        state = tf.zeros(shape=(B, H, DK, DV), dtype="f32")
        gf = tf.cast(g[:, 0, :], dtype="f32")
        gf = gf * tf.full_like(gf, value=0.01)
        alpha = tf.exp(gf)
        kf = tf.reshape(tf.cast(k[:, 0, :, :], dtype="f32"), new_shape=(B, H, DK))
        kf = kf * tf.full_like(kf, value=0.01)
        vf = tf.reshape(tf.cast(v[:, 0, :, :], dtype="f32"), new_shape=(B, H, DV))
        vf = vf * tf.full_like(vf, value=0.01)
        betaf = tf.reshape(tf.cast(beta[:, 0, :], dtype="f32"), new_shape=(B, H, 1))
        betaf = betaf * tf.full_like(betaf, value=0.01)
        for t in range(S):
            old = tf.reshape(
                tf.matmul(tf.reshape(kf, new_shape=(B, H, 1, DK)), state),
                new_shape=(B, H, DV),
            )
            delta = betaf * (vf - tf.reshape(alpha, new_shape=(B, H, 1)) * old)
            state = (
                tf.reshape(alpha, new_shape=(B, H, 1, 1)) * state
                + tf.reshape(kf, new_shape=(B, H, DK, 1))
                * tf.reshape(delta, new_shape=(B, H, 1, DV))
            )
        return tf.cast(state, dtype=DT)
