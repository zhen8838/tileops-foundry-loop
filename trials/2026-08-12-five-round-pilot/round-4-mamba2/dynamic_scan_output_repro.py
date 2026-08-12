"""Minimal HIR repro: insert each ordered-loop value at its induction offset."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- bindings used by @func


SEQLEN = 4


@module(entry="scan_copy")
class ScanCopy:
    @func
    def scan_copy(x: Tensor[(SEQLEN, 4), "f32"]):
        out = tf.zeros(shape=(SEQLEN, 4), dtype="f32")
        for t in tile(SEQLEN, 1):
            out = tf.insert_slice(out, x[t, :], (t, 0))
        return out
