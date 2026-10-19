"""mHC stream read / write through DeepSeek's TileKernels (tile_kernels.mhc), in Megatron's layout.

Megatron's mHC (megatron/core/transformer/mhc.py) keeps the residual as ``[s, b, n, C]`` streams:

* read:  ``x = sum_j h_pre[j] * streams[j]``                        (TileKernels: pre_apply_mix)
* write: ``streams'[i] = sum_j h_res[i, j] streams[j] + h_post[i] f`` (TileKernels: post)

TileKernels indexes the stream mixer as ``comb[source, target]``; Megatron's ``h_res`` is
``[target, source]``, so the write passes ``h_res^T``. Coefficients stay fp32, streams bf16. The
Sinkhorn projection is NOT taken from TileKernels: theirs is the exp / normalize form whose
clamped sums gave NaN gradients under MoE (Megatron uses the log-space form).
"""

from .ops import mhc_read, mhc_write

__all__ = ["mhc_read", "mhc_write"]
