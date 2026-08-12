"""Typed full-sequence Mamba-2 recurrence and output insertion reproducer."""

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


@module(entry="mamba2_fwd")
class Mamba2Fwd:
    @func
    def mamba2_fwd(
        x: Tensor[(BATCH, SEQLEN, HEADS, HEAD_DIM), DTYPE],
        dt: Tensor[(BATCH, SEQLEN, HEADS), "f32"],
        A: Tensor[(HEADS,), "f32"],
        B: Tensor[(BATCH, SEQLEN, GROUPS, STATE_DIM), DTYPE],
        C: Tensor[(BATCH, SEQLEN, GROUPS, STATE_DIM), DTYPE],
    ):
        state = tf.zeros(
            shape=(BATCH, HEADS, HEAD_DIM, STATE_DIM), dtype="f32"
        )
        y = tf.zeros(shape=(BATCH, SEQLEN, HEADS, HEAD_DIM), dtype="f32")
        for t in tile(SEQLEN, 1):
            delta = tf.clamp(tf.softplus(dt[:, t, :]), min_val=0.0, max_val=MAX_DT)
            decay = tf.exp(delta * tf.reshape(A, new_shape=(1, HEADS)))
            B_heads = tf.repeat_interleave(B[:, t, :, :], repeats=HEADS, axis=1)
            C_heads = tf.repeat_interleave(C[:, t, :, :], repeats=HEADS, axis=1)
            update = (
                tf.reshape(delta, new_shape=(BATCH, HEADS, 1, 1))
                * tf.reshape(
                    tf.cast(x[:, t, :, :], dtype="f32"),
                    new_shape=(BATCH, HEADS, HEAD_DIM, 1),
                )
                * tf.reshape(
                    tf.cast(B_heads, dtype="f32"),
                    new_shape=(BATCH, HEADS, 1, STATE_DIM),
                )
            )
            state = (
                tf.reshape(decay, new_shape=(BATCH, HEADS, 1, 1)) * state
                + update
            )
            y_t = tf.reduce(
                state
                * tf.reshape(
                    tf.cast(C_heads, dtype="f32"),
                    new_shape=(BATCH, HEADS, 1, STATE_DIM),
                ),
                axes=(-1,),
                keepdim=False,
                kind="sum",
            )
            y = tf.insert_slice(
                y,
                tf.reshape(y_t, new_shape=(BATCH, 1, HEADS, HEAD_DIM)),
                (0, t, 0, 0),
            )
        return y, state
