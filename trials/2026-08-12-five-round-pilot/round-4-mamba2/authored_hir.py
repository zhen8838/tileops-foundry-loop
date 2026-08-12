"""Authored HIR for one Mamba-2 primary-contract recurrence step.

The fixed fixture uses G=1, as do both primary manifest rows. Repeating the
single group across heads is the static form of group(h) = floor(h / (H/G)).
All recurrence arithmetic and both outputs remain f32.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bindings used by @func
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


BATCH = 1
HEADS = 2
HEAD_DIM = 16
GROUPS = 1
STATE_DIM = 16
DTYPE = "f16"
MAX_DT = float("inf")


@module(
    entry="mamba2_step",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 512)),
)
class Mamba2Step:
    @func
    def mamba2_step(
        state: Tensor[(BATCH, HEADS, HEAD_DIM, STATE_DIM), "f32"],
        x_t: Tensor[(BATCH, HEADS, HEAD_DIM), DTYPE],
        dt_t: Tensor[(BATCH, HEADS), "f32"],
        A: Tensor[(HEADS,), "f32"],
        B_t: Tensor[(BATCH, GROUPS, STATE_DIM), DTYPE],
        C_t: Tensor[(BATCH, GROUPS, STATE_DIM), DTYPE],
    ):
        delta = tf.clamp(tf.softplus(dt_t), min_val=0.0, max_val=MAX_DT)
        decay = tf.exp(delta * tf.reshape(A, new_shape=(1, HEADS)))
        B_heads = tf.repeat_interleave(B_t, repeats=HEADS, axis=1)
        C_heads = tf.repeat_interleave(C_t, repeats=HEADS, axis=1)
        update = (
            tf.reshape(delta, new_shape=(BATCH, HEADS, 1, 1))
            * tf.reshape(tf.cast(x_t, dtype="f32"), new_shape=(BATCH, HEADS, HEAD_DIM, 1))
            * tf.reshape(
                tf.cast(B_heads, dtype="f32"),
                new_shape=(BATCH, HEADS, 1, STATE_DIM),
            )
        )
        next_state = (
            tf.reshape(decay, new_shape=(BATCH, HEADS, 1, 1)) * state + update
        )
        y_t = tf.reduce(
            next_state
            * tf.reshape(
                tf.cast(C_heads, dtype="f32"),
                new_shape=(BATCH, HEADS, 1, STATE_DIM),
            ),
            axes=(-1,),
            keepdim=False,
            kind="sum",
        )
        return y_t, next_state
