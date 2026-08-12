"""Authored HIR for representable complex64 FFT building blocks.

Complex numbers are explicit f32 real/imaginary pairs. Twiddles are inputs, so
the graph states the FFT arithmetic without relying on trigonometric authoring.
The DFT uses W[t, k] orientation and therefore emits natural-frequency order.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


BATCH = 2
N = 4
HALF = 2


@module(
    entry="dft_pair_f32",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 256)),
)
class FFTPairF32:
    @func
    def complex_butterfly(
        even_r: Tensor[(BATCH, HALF), "f32"],
        even_i: Tensor[(BATCH, HALF), "f32"],
        odd_r: Tensor[(BATCH, HALF), "f32"],
        odd_i: Tensor[(BATCH, HALF), "f32"],
        tw_r: Tensor[(HALF,), "f32"],
        tw_i: Tensor[(HALF,), "f32"],
    ):
        prod_r = tf.sub(tf.mul(odd_r, tw_r), tf.mul(odd_i, tw_i))
        prod_i = tf.add(tf.mul(odd_r, tw_i), tf.mul(odd_i, tw_r))
        out_r = tf.concat(tf.add(even_r, prod_r), tf.sub(even_r, prod_r), axis=1)
        out_i = tf.concat(tf.add(even_i, prod_i), tf.sub(even_i, prod_i), axis=1)
        return out_r, out_i

    @func
    def dft_pair_f32(
        x_r: Tensor[(BATCH, N), "f32"],
        x_i: Tensor[(BATCH, N), "f32"],
        w_r: Tensor[(N, N), "f32"],
        w_i: Tensor[(N, N), "f32"],
    ):
        y_r = tf.sub(tf.matmul(x_r, w_r), tf.matmul(x_i, w_i))
        y_i = tf.add(tf.matmul(x_r, w_i), tf.matmul(x_i, w_r))
        return y_r, y_i
