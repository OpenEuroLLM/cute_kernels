# OpenEuroLLM addition to the vendored kernels (same style and helpers as window_end.py).
"""The reads of a window's edges from the running state, for layers that are run again under
activation recompute.

In the backward the running state M is at a window boundary: after the window's last edge (the
window's end state T, before its window-end launch) or before its first (the start state B, after
it). The read of edge j of that window is then, with the window's gains kappa_i, queries q_j and
residuals u_i (the tapes),

    from the end:    r_j = T^T q_j - sum_{i > j} (kappa_i . q_j) u_i
    from the start:  r_j = B^T q_j + sum_{i <= j} (kappa_i . q_j) u_i

which are exact rearrangements of M_j^T q_j with M_j = T - sum_{i>j} kappa_i u_i^T, the same sums
the forward kernel takes for its reads. One pass over M per launch; the local edges [lo, hi) are
written to ``reads`` (rows of E) in BF16. One warp per group, every lane ``width`` columns.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, cute

from .window_end import _load_dense, _store_bf16_vector, default_lane_width


class RecomputeRead:
    def __init__(self, E, N, G, K, D, *, sms, lo, hi, from_end, warps=4, blocks_per_sm=4,
                 lane_width=None):
        self.E, self.N, self.G, self.K, self.D = E, N, G, K, D
        self.lo, self.hi, self.from_end = lo, hi, from_end
        self.warps = warps
        self.blocks = min((N * G + warps - 1) // warps, blocks_per_sm * sms)
        self.lane_width = default_lane_width(D) if lane_width is None else lane_width
        self.shared_bytes = 0
        self.static_edge = None

    @cute.jit
    def __call__(self, m: cute.Tensor, queries: cute.Tensor, gains: cute.Tensor,
                 residuals: cute.Tensor, reads: cute.Tensor, stream: cuda.CUstream):
        for tensor, shape, dtype in (
            (m, (self.N, self.G, self.K, self.D), BFloat16),
            (queries, (self.E, self.N, self.G, self.K), Float32),
            (gains, (self.E, self.N, self.G, self.K), Float32),
            (residuals, (self.E, self.N, self.G, self.D), BFloat16),
            (reads, (self.E, self.N, self.G, self.D), BFloat16),
        ):
            assert tensor.element_type == dtype
            assert tuple(tensor.shape) == shape
            stride = 1
            for axis in cutlass.range_constexpr(len(shape) - 1, -1, -1):
                assert tensor.stride[axis] == stride
                stride *= shape[axis]
        self.kernel(m, queries, gains, residuals, reads).launch(
            grid=(self.blocks, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, m: cute.Tensor, queries: cute.Tensor, gains: cute.Tensor,
               residuals: cute.Tensor, reads: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        K, D, E = self.K, self.D, self.E
        lo: cutlass.Constexpr = self.lo
        hi: cutlass.Constexpr = self.hi
        width = math.gcd(D, self.lane_width, min(
            m.iterator.max_alignment, residuals.iterator.max_alignment,
            reads.iterator.max_alignment) // 2)
        ng = self.N * self.G
        full: cutlass.Constexpr = D % (32 * width) == 0
        stride = Int32(self.blocks * self.warps)
        for group in cutlass.range(bid * self.warps + warp, ng, stride):
            n, g = group // self.G, group % self.G
            # The coefficients kappa_i . q_j of the window, and the queries, per lane in registers.
            q = cute.make_rmem_tensor((hi - lo, K), Float32)
            coef = cute.make_rmem_tensor((hi - lo, E), Float32)
            coef.fill(0.0)
            for j in cutlass.range_constexpr(lo, hi):
                for k in cutlass.range_constexpr(K):
                    q[j - lo, k] = queries[j, n, g, k]
                for i in cutlass.range_constexpr(E):
                    uses: cutlass.Constexpr = (i > j) if self.from_end else (i <= j)
                    if cutlass.const_expr(uses):
                        value = Float32(0.0)
                        for k in cutlass.range_constexpr(K):
                            value += gains[i, n, g, k] * q[j - lo, k]
                        coef[j - lo, i] = -value if cutlass.const_expr(self.from_end) else value
            for tile in cutlass.range_constexpr(cute.ceil_div(D, 32 * width)):
                d = Int32(tile) * (32 * width) + lane * width
                inside = (d < D) | Boolean(full)
                if inside:
                    state = cute.make_rmem_tensor((K, width), Float32)
                    for k in cutlass.range_constexpr(K):
                        loaded = _load_dense(m, (Int64(group) * K + k) * D + Int64(d), width)
                        for v in cutlass.range_constexpr(width):
                            state[k, v] = loaded[v]
                    residual = cute.make_rmem_tensor((E, width), Float32)
                    for i in cutlass.range_constexpr(E):
                        loaded = _load_dense(residuals, (Int64(i) * ng + group) * D + Int64(d), width)
                        for v in cutlass.range_constexpr(width):
                            residual[i, v] = loaded[v]
                    for j in cutlass.range_constexpr(lo, hi):
                        read = cute.make_rmem_tensor(width, Float32)
                        read.fill(0.0)
                        for k in cutlass.range_constexpr(K):
                            for v in cutlass.range_constexpr(width):
                                read[v] += state[k, v] * q[j - lo, k]
                        for i in cutlass.range_constexpr(E):
                            uses: cutlass.Constexpr = (i > j) if self.from_end else (i <= j)
                            if cutlass.const_expr(uses):
                                for v in cutlass.range_constexpr(width):
                                    read[v] += coef[j - lo, i] * residual[i, v]
                        _store_bf16_vector(reads, (Int64(j) * ng + group) * D + Int64(d), read, width)


def make_read(E: int, N: int, G: int, K: int, D: int, *, sms: int, lo: int, hi: int,
              from_end: bool, warps: int = 4, blocks_per_sm: int = 4, lane_width=None):
    """The read kernel for the local edges [lo, hi) of a window of E edges, from the window's end
    state (``from_end``) or its start state."""
    if not 0 <= lo < hi <= E:
        raise ValueError("the read kernel needs 0 <= lo < hi <= E")
    if K not in (4, 8, 16) or D % 8:
        raise ValueError("the read kernel requires K in {4, 8, 16} and D % 8 == 0")
    if type(from_end) is not bool:
        raise ValueError("from_end must be a Python bool")
    return RecomputeRead(E, N, G, K, D, sms=sms, lo=lo, hi=hi, from_end=from_end, warps=warps,
                         blocks_per_sm=blocks_per_sm, lane_width=lane_width)
