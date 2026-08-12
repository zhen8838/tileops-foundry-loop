"""Executable HIR proof of the ordered FP32 Mamba-2 carry."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bindings used by @func


BATCH = 1
SEQLEN = 4
HEADS = 2
HEAD_DIM = 16
GROUPS = 1
STATE_DIM = 16
DTYPE = "f16"
MAX_DT = float("inf")


@module(entry="mamba2_state_scan")
class Mamba2StateScan:
    @func
    def mamba2_state_scan(
        x: Tensor[(BATCH, SEQLEN, HEADS, HEAD_DIM), DTYPE],
        dt: Tensor[(BATCH, SEQLEN, HEADS), "f32"],
        A: Tensor[(HEADS,), "f32"],
        B: Tensor[(BATCH, SEQLEN, GROUPS, STATE_DIM), DTYPE],
    ):
        state = tf.zeros(
            shape=(BATCH, HEADS, HEAD_DIM, STATE_DIM), dtype="f32"
        )
        # Fixed token zero isolates ordered loop carry from the separately
        # demonstrated dynamic token-index/output-insertion parser boundary.
        delta = tf.clamp(tf.softplus(dt[:, 0, :]), min_val=0.0, max_val=MAX_DT)
        decay = tf.exp(delta * tf.reshape(A, new_shape=(1, HEADS)))
        B_heads = tf.repeat_interleave(B[:, 0, :, :], repeats=HEADS, axis=1)
        update = (
            tf.reshape(delta, new_shape=(BATCH, HEADS, 1, 1))
            * tf.reshape(
                tf.cast(x[:, 0, :, :], dtype="f32"),
                new_shape=(BATCH, HEADS, HEAD_DIM, 1),
            )
            * tf.reshape(
                tf.cast(B_heads, dtype="f32"),
                new_shape=(BATCH, HEADS, 1, STATE_DIM),
            )
        )
        for _ in range(SEQLEN):
            state = (
                tf.reshape(decay, new_shape=(BATCH, HEADS, 1, 1)) * state
                + update
            )
        return state
