# Launch helpers vendored from github.com/davidgonmar/depth-memory (commit 9a6865f, MIT License,
# Copyright (c) 2026 David Gonzalez Martinez), olmo_core/nn/ldm/backends/fast/state.py; the
# workspace class is ours (OpenEuroLLM), adapted from that file's SegmentState.

"""The LDM state of the paper's CuTe kernels as a static, CUDA-graph capturable workspace.

Interface (shared with Megatron's ``TorchDepthMemoryWorkspace``): ``ensure(n, device)``,
``forward(edge, khat, query, value, scale, forgetting) -> read`` and ``backward(edge, read_grad)
-> (d khat, d query, d value, d scale, d forgetting [G])``, the backward in strictly decreasing
edge order. Differences to SegmentState: all buffers are allocated once per token count and kept
(a CUDA graph bakes their addresses), the state resets at edge 0 of every forward, the backward
starts at the last edge, and the window bookkeeping is a function of the edge alone (no graph-task
tracking, no release callback, no record_stream) -- so a forward / backward is the same sequence
of launches on the same memory every micro-batch, and per-layer graphs can capture them.
"""

from __future__ import annotations

import gc
import math
import os
from functools import lru_cache

import torch

from . import schedules



@lru_cache(maxsize=256)
def _program(kernel: str, options: tuple, sizes: tuple[int, ...], metadata: tuple,
             capability: tuple[int, int], device_index: int, schedule: tuple[int, ...],
             edge: int | None):
    """The compiled ``kernel`` ("forward", "interior" or "window_end") for one option set, window
    size, operand layout and launch schedule on this device. Every kernel takes its tensors, then
    the window-local edge (except the window end), then the stream. Only compiled code and
    metadata are cached, never tensors."""
    # CUDA module handles are context-local, so the device is part of the key; its SM count
    # and shared-memory budget only size the persistent grids.
    properties = torch.cuda.get_device_properties(device_index)
    del device_index
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    types = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16,
             torch.float32: cutlass.Float32}
    fake = [None if item is None else make_fake_tensor(
        types[item[0]], item[1], item[2], assumed_align=item[3]) for item in metadata]
    if kernel in ("forward", "interior"):
        fake.append(cutlass.Int32(0))            # the window-local edge
    fake.append(make_fake_stream(use_tvm_ffi_env_stream=False))
    arch = {(9, 0): "sm_90a", (10, 0): "sm_100a"}.get(capability, f"sm_{capability[0]}{capability[1]}")
    operation = schedules.build_operation(kernel, sizes, options, properties, schedule)
    if edge is not None:
        # The window-local edge is compiled in; the kernel still takes it as an argument.
        operation.static_edge = edge
    return cute.compile(operation, *fake, options=f"--gpu-arch={arch}")


def _metadata(tensors: list[torch.Tensor | None]):
    """(device, per-operand dtype, shape, strides and 16-byte alignment) of one launch."""
    device = None
    metadata = []
    for tensor in tensors:
        if tensor is None:
            metadata.append(None)
            continue
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError("kernel launch tensors must share one device")
        # Alignment depends on the address (of views in particular), so it is part of every
        # launch's metadata and of the compiled program's key.
        metadata.append((tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()),
                         math.gcd(tensor.data_ptr(), 16)))
    if device is None:
        raise ValueError("a kernel launch requires at least one tensor")
    return device, tuple(metadata)


def _run(kernel, sizes, tensors, metadata, edge, options, schedule):
    """Launch ``kernel`` with an explicit schedule on the current stream."""
    device_index = torch.cuda.current_device()
    program = _program(kernel, options, sizes, metadata, torch.cuda.get_device_capability(device_index),
                       device_index, schedule, edge)
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    # Views can keep requires_grad inside Function.forward, so detach what we hand to the kernel.
    # The autograd graph itself is untouched. (cute_ldm: views of the static workspace buffers are
    # converted once and cached -- the same memory every micro-batch; per-call tensors each time.)
    arguments = []
    for tensor, item in zip(tensors, metadata, strict=True):
        if tensor is None:
            arguments.append(None)
            continue
        key = None
        if tensor.untyped_storage().data_ptr() in _STATIC_STORAGES:
            key = (tensor.data_ptr(), item[1], item[2], item[3], item[0])
            cached = _STATIC_ARGS.get(key)
            if cached is not None:
                arguments.append(cached)
                continue
        converted = from_dlpack(tensor.detach(), assumed_align=item[3])
        if key is not None:
            _STATIC_ARGS[key] = converted
        arguments.append(converted)
    if edge is not None:
        arguments.append(cutlass.Int32(edge))
    arguments.append(cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    program(*arguments)


def _launch(kernel: str, sizes: tuple[int, ...], tensors: list[torch.Tensor | None], edge: int | None = None,
            *, options: tuple = (), training: bool):
    """One kernel launch with the schedule its GPU's table names for this shape."""
    device, metadata = _metadata(tensors)
    with torch.cuda.device(device):
        # Compiling a kernel lets the CuTe DSL keep the calling Python frames alive (its
        # decorators store inspect.currentframe()), and with them the launch's tensors. Only the
        # cyclic collector frees them, and the trainer runs it rarely. So after a compile, collect
        # once at the next launch, when those frames are gone.
        misses = _program.cache_info().misses
        if misses != _launch_state["misses"]:
            gc.collect()
            _launch_state["misses"] = misses
        # (cute_ldm: no record_stream -- the workspace buffers are static, never freed)
        schedule = schedules.schedule(kernel, sizes, options, edge, device, training=training)
        _run(kernel, sizes, tensors, metadata, edge, options, schedule)


_launch_state = {"misses": 0}
# storages of the live workspaces' buffers, and their launch arguments (see _run)
_STATIC_STORAGES = set()
_STATIC_ARGS = {}

# The dtypes the forward may keep the running covariance in.
RUNNING_COVARIANCE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
# The 16-bit types of M, dM, the residual / read / DY / DV tapes and the value gradients: BF16, or
# FP16 (the kernels take it from their operands) -- for FP16 training, or for BF16 training with
# ``grad_scaling``.
STORAGE_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
FP16_MAX = 65504.0
# grad_scaling: the largest |cotangent| after scaling. FP16 keeps 2^11 above it for cotangents that
# outgrow the scale within a backward and for dM, which sums ~2 |q| |dy| per edge over the depth.
GRAD_SCALE_TARGET = 2.0 ** 5


def _scaled_cotangent(grad, scale):
    """grad_scaling: (grad * scale in FP16, saturated at its range; max |grad|)."""
    grad = grad.float()
    return (grad * scale).clamp(-FP16_MAX, FP16_MAX).to(torch.float16), grad.abs().amax()


def _unscaled(grads, scale):
    """grad_scaling: every gradient / scale in FP32 with its non-finite entries zeroed, and how
    many were."""
    inverse, out, bad = scale.reciprocal(), [], 0
    for grad in grads:
        grad = grad.float() * inverse
        finite = torch.isfinite(grad)
        bad = bad + finite.logical_not().sum()
        out.append(torch.where(finite, grad, 0.0))
    return out, bad


_FUSED = {"scaled_cotangent": _scaled_cotangent, "unscaled": _unscaled}
_COMPILED = {}


def _fused(name):
    """``name``'s function compiled (Inductor fuses its elementwise passes and reductions into
    one or two kernels per tensor); eager with CUTE_LDM_EAGER_SCALING=1 or if compiling fails."""
    if os.environ.get("CUTE_LDM_EAGER_SCALING") == "1":
        return _FUSED[name]
    if name not in _COMPILED:
        try:
            _COMPILED[name] = torch.compile(_FUSED[name], dynamic=False)
        except Exception:  # pragma: no cover - Inductor unavailable
            _COMPILED[name] = _FUSED[name]
    return _COMPILED[name]


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` if it is contiguous and 16-byte aligned, else a contiguous copy."""
    if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


class CuteDepthMemoryWorkspace:
    """The tapes and buffers of ``total_edges`` LDM steps for ``n`` tokens (see module doc).

    ``grad_scaling``: FP16 storage under BF16 (or FP32) training. FP16 holds the backward's
    gradient tapes (dM, DY, DV, the value gradient) as well as M, and unscaled activation
    gradients (~1e-7 and below) underflow it. Every backward therefore scales its cotangents by
    a power of two, ``grad_scale``, and divides the input gradients by it (FP32 out); the backward
    is linear in the cotangents, so this is exact up to rounding. The scale maps the larger of the
    previous backward's largest |cotangent| and this backward's first one to GRAD_SCALE_TARGET.
    Scaled cotangents saturate at FP16's range; non-finite input gradients are zeroed, counted in
    ``grad_nonfinite``, and make the next backward's scale 16x smaller. FP16 training brings
    its own loss scale and leaves this off.
    """

    def __init__(self, *, total_edges: int, groups: int, key_dim: int, value_dim: int,
                 alpha: float = 1.0, span: int = 4, running_covariance_dtype: str = "float32",
                 storage_dtype: str = "bfloat16", grad_scaling: bool = False):
        if key_dim not in (4, 8, 16):
            raise ValueError("cute_ldm: the key dimension must be 4, 8 or 16")
        if running_covariance_dtype != "float32":
            raise ValueError("cute_ldm: the running covariance must be float32 (the backward "
                             "unwinds P from it; a BF16 P drifts indefinite over deep stacks)")
        self.total_edges, self.span = total_edges, min(span, total_edges)
        self.groups, self.key_dim, self.value_dim = groups, key_dim, value_dim
        self.padded_value_dim = -(-value_dim // 8) * 8
        self.covariance_scale = float(alpha)
        self.running_covariance_dtype = running_covariance_dtype
        if storage_dtype not in STORAGE_DTYPES:
            raise ValueError(f"cute_ldm: storage_dtype must be one of {sorted(STORAGE_DTYPES)}")
        self.storage = STORAGE_DTYPES[storage_dtype]
        if grad_scaling and self.storage != torch.float16:
            raise ValueError("cute_ldm: grad_scaling is for FP16 storage")
        self.grad_scaling = bool(grad_scaling)
        self.n = None

    # -- buffers -------------------------------------------------------------------------------
    def ensure(self, n: int, device) -> None:
        if self.n == n and self.m.device == torch.device(device):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cute_ldm: workspace allocation inside a CUDA graph capture")
        self.n, self.device = n, torch.device(device)
        self.state_loaded = False  # load_state: this segment starts from a carried state
        self.grad_loaded = False  # load_grad: the backward starts from a carried state gradient
        g, k, d, length, span = self.groups, self.key_dim, self.padded_value_dim, \
            self.total_edges, self.span

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        self.m = empty((n, g, k, d), self.storage)
        dtype = RUNNING_COVARIANCE_DTYPES[self.running_covariance_dtype]
        per_16_bytes = 16 // dtype.itemsize
        triangle = k * (k + 1) // 2
        packed = torch.zeros((-(-triangle // per_16_bytes) * per_16_bytes,), dtype=dtype,
                             device=self.device)
        row = torch.arange(k, device=self.device)
        packed.index_fill_(0, row * k - row * (row - 1) // 2, self.covariance_scale)
        self.p0 = packed.expand(n, g, packed.numel()).clone()
        self.p = self.p0.clone()
        self._allocate_tapes(n)
        rows = self._backward_rows()
        # the window's pre-write covariances, FP32, unwound by its window-end launch for its
        # interior launches
        self.p_scratch = empty((rows, n, g, triangle))
        self.dm = empty((n, g, k, d), self.storage)
        self.covariance_grad = empty((n, g, triangle))
        self.future_value_grad = empty((rows, n, g, d))
        self.dy, self.dv = [empty((rows, n, g, d), self.storage) for _ in range(2)]
        self.future_gain_grad = empty((rows, n, g, k))
        # grad_scaling: this backward's scale, its largest |cotangent| so far, the previous
        # backward's, and the non-finite input gradients zeroed so far (all on the device: no sync)
        self.grad_scale = torch.ones((), device=self.device)
        self.grad_amax = torch.zeros((), device=self.device)
        self.grad_amax_prev = torch.zeros((), device=self.device)
        self.grad_nonfinite = torch.zeros((), dtype=torch.int64, device=self.device)
        self._nonfinite_before = torch.zeros((), dtype=torch.int64, device=self.device)
        self._allocate_reads(n)
        self._register_buffers()

    def _backward_rows(self) -> int:
        """The most edges in a backward window: the rows of the backward's work buffers."""
        return self.span

    def _allocate_reads(self, n: int) -> None:
        self.reads = torch.empty((self.span, n, self.groups, self.padded_value_dim),
                                 dtype=self.storage, device=self.device)   # recompute_reads output

    def _allocate_tapes(self, n: int) -> None:
        """The per-edge tapes, [edge, n, g, ...] over every edge."""
        g, k, d, length = self.groups, self.key_dim, self.padded_value_dim, self.total_edges

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        self.keys, self.queries, self.gains = [empty((length, n, g, k)) for _ in range(3)]
        self.residuals = empty((length, n, g, d), self.storage)
        self.denominators, self.forgetting_tape, self.scales = [empty((length, n, g))
                                                                for _ in range(3)]

    def _register_buffers(self) -> None:
        # the launch-argument cache of these buffers (their addresses are fixed from here on);
        # a reallocation drops the previous buffers' entries first
        _STATIC_ARGS.clear()
        _STATIC_STORAGES.difference_update(getattr(self, "_storages", ()))
        self._storages = {value.untyped_storage().data_ptr() for value in vars(self).values()
                          if isinstance(value, torch.Tensor)}
        _STATIC_STORAGES.update(self._storages)

    def _padded(self, rows):
        padding = self.padded_value_dim - self.value_dim
        return torch.nn.functional.pad(rows, (0, padding)) if padding else rows

    def _unpadded(self, rows):
        return rows[..., :self.value_dim] if self.padded_value_dim != self.value_dim else rows

    def _window(self, edge):
        start = edge // self.span * self.span
        return start, min(start + self.span, self.total_edges)

    # -- storage types and the gradient scale (grad_scaling) ----------------------------------
    def _to_storage(self, value):
        """A written value in the storage type; FP16 from a wider type saturates at its largest
        finite value instead of overflowing to inf."""
        if self.storage == torch.float16 and value.dtype != torch.float16:
            value = value.clamp(-FP16_MAX, FP16_MAX)
        return value.to(self.storage)

    def _begin_backward(self, read_grad) -> None:
        """grad_scaling, at the backward's first edge: this backward's scale."""
        if not self.grad_scaling:
            return
        amax = torch.maximum(read_grad.detach().abs().amax().float(), self.grad_amax_prev)
        exponent = torch.floor(torch.log2(GRAD_SCALE_TARGET / amax.clamp_min(1e-30))).clamp(-100, 100)
        self.grad_scale.copy_(torch.where(amax > 0, torch.exp2(exponent), torch.ones_like(amax)))
        self.grad_amax.zero_()
        self._nonfinite_before.copy_(self.grad_nonfinite)

    def _end_backward(self) -> None:
        """grad_scaling, after the backward's last edge (edge 0): the next backward's reference
        amax, 16x larger (a 16x smaller scale) if this backward zeroed non-finite gradients."""
        if not self.grad_scaling:
            return
        overflowed = self.grad_nonfinite > self._nonfinite_before
        self.grad_amax_prev.copy_(torch.where(overflowed, self.grad_amax * 16, self.grad_amax))

    def _cotangent(self, read_grad):
        """A read's cotangent as the backward's 16-bit tape (scaled under grad_scaling)."""
        if not self.grad_scaling:
            return read_grad.to(self.storage)
        scaled, amax = _fused("scaled_cotangent")(read_grad.detach(), self.grad_scale)
        torch.maximum(self.grad_amax, amax, out=self.grad_amax)
        return scaled

    def _input_grads(self, raw_key_grad, query_grad, raw_value_grad, scale_grad, forgetting_grad):
        """The step's input gradients; under grad_scaling unscaled to FP32 (an unscaled FP16 value
        gradient would underflow), non-finite entries zeroed and counted."""
        grads = (raw_key_grad, query_grad, raw_value_grad, scale_grad, forgetting_grad)
        if not self.grad_scaling:
            return grads
        out, bad = _fused("unscaled")(grads, self.grad_scale)
        self.grad_nonfinite += bad
        return tuple(out)

    def numerics_stats(self) -> dict:
        """grad_scaling: the last backward's scale and largest |cotangent|, and the non-finite
        input gradients zeroed so far (device tensors)."""
        return dict(grad_scale=self.grad_scale, grad_amax=self.grad_amax_prev,
                    grad_nonfinite=self.grad_nonfinite)

    # -- forward -------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, edge, khat, query, value, scale, forgetting) -> torch.Tensor:
        n, g, k = self.n, self.groups, self.key_dim
        if edge == 0 and not self.state_loaded:
            self.p.copy_(self.p0)
        start, end = self._window(edge)
        zero_start = start == 0 and not self.state_loaded
        raw_key = _aligned(khat.float().reshape(n, g, k))
        query = _aligned(query.float().reshape(n, g, k))
        scale = _aligned(scale.float().reshape(n, g))
        forgetting = torch.broadcast_to(forgetting.float(), (n, g))
        if forgetting.stride() != (0, 1):
            forgetting = _aligned(forgetting)
        value = self._padded(self._to_storage(value).contiguous().view(n, g, self.value_dim))
        read = torch.empty_like(value)
        tapes = [self.keys[edge], self.queries[edge], self.scales[edge], self.forgetting_tape[edge]]
        sizes = (end - start, n, g, k, self.padded_value_dim)
        options = (("first_window", zero_start),
                   ("running_covariance_dtype", self.running_covariance_dtype))
        _launch("forward", sizes,
                [self.m, self.p, raw_key, query, value, scale, forgetting, self.gains[start:end],
                 self.residuals[start:end], self.denominators[start:end], read, *tapes],
                edge - start, options=options, training=True)
        return self._unpadded(read)

    # -- backward ------------------------------------------------------------------------------
    @torch.no_grad()
    def backward(self, edge, read_grad):
        if edge == self.total_edges - 1 and not self.grad_loaded:
            self.dm.zero_()
            self.covariance_grad.zero_()
            self._begin_backward(read_grad)
        start, end = self._window(edge)
        zero_start = start == 0 and not self.state_loaded
        count = end - start
        scratch = self.p_scratch[:count]
        n, g, k, d = self.n, self.groups, self.key_dim, self.padded_value_dim
        query_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        forgetting_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        query_output = query_grad.expand(count, n, g, k)
        forgetting_output = forgetting_grad.expand(count, n, g)
        raw_value_grad = torch.empty((n, g, d), device=self.device, dtype=self.storage)
        raw_key_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        scale_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        current_dy = _aligned(self._padded(
            self._cotangent(read_grad).reshape(n, g, self.value_dim)))
        if edge == end - 1:
            # the window end; its future window is the next one (processed just before)
            fs, fe = end, min(end + self.span, self.total_edges)
            future = fe - fs
            if future == 0:
                fs, fe = start, start + 1  # F=0 placeholders the window end never reads
            _launch("window_end", (count, future, n, g, k, d),
                    [self.m, self.dm, self.keys[start:end], self.queries[start:end],
                     self.gains[start:end], self.residuals[start:end],
                     self.forgetting_tape[start:end], self.denominators[start:end],
                     self.queries[fs:fe], self.keys[fs:fe],
                     self.dy[:max(1, future)], self.dv[:max(1, future)], self.m, self.dm,
                     self.future_value_grad[:count], self.future_gain_grad[:count], current_dy,
                     self.scales[edge], self.dy[:count], self.dv[:count], query_output,
                     self.covariance_grad, forgetting_output,
                     raw_value_grad, raw_key_grad, scale_grad, self.p, scratch],
                    options=(("first_window", zero_start),), training=True)
        else:
            _launch("interior", (count, n, g, k, d),
                    [self.m, self.residuals[start:end], self.dy[:count], self.keys[start:end],
                     self.gains[start:end], self.queries[start:end],
                     self.future_value_grad[:count], self.future_gain_grad[:count],
                     self.dv[:count], query_output, self.covariance_grad,
                     self.forgetting_tape[start:end], self.denominators[start:end],
                     forgetting_output, self.scales[edge], raw_value_grad, raw_key_grad,
                     scale_grad, scratch, current_dy], edge - start, training=True)
        if edge == 0:
            self._end_backward()
        return self._input_grads(raw_key_grad, query_grad, self._unpadded(raw_value_grad),
                                 scale_grad, forgetting_grad.sum(0))

    # -- reads under activation recompute ------------------------------------------------------
    @torch.no_grad()
    def recompute_reads(self, start, end, lo, hi, from_end: bool) -> torch.Tensor:
        """The reads of the local edges [lo, hi) of the window [start, end) from the running state,
        which is at that window's end (``from_end``) or its start; [hi - lo, n, G, D] in the
        storage type (padded width sliced off)."""
        n, g, k, d = self.n, self.groups, self.key_dim, self.padded_value_dim
        count = end - start
        _launch("read", (count, n, g, k, d),
                [self.m, self.queries[start:end], self.gains[start:end], self.residuals[start:end],
                 self.reads[:count]],
                options=(("lo", lo), ("hi", hi), ("from_end", from_end)), training=True)
        return self._unpadded(self.reads[lo:hi])

    # -- segments (a state carried across recompute chunks / pipeline stages) ------------------
    # A segment is the edges of one recompute chunk or stage: its workspace (``total_edges`` = the
    # segment's edges, windows from its first edge) starts from the previous segment's state, and
    # its backward from the next segment's state gradient. M is committed at every window end, so
    # the state at a window boundary is complete. The backward is not: a window's own read /
    # residual terms of dL/dM_start are added by the PREVIOUS window's window-end launch (its
    # "future" tapes), so at a segment start they are folded into dM here (export_grad) and the
    # previous segment's last window runs with no future, from the carried dM.
    def load_state(self, m: torch.Tensor, p: torch.Tensor) -> None:
        """Start the next forward from M ``[n, G, K, D]`` (storage type) and the packed running P."""
        self.m.copy_(m.view_as(self.m))
        self.p.copy_(p.view_as(self.p))
        self.state_loaded = True

    def export_state(self):
        """(M, P) after the forward's last edge (views of the workspace: copy before reuse)."""
        return self.m, self.p

    def load_grad(self, dm: torch.Tensor, covariance_grad: torch.Tensor) -> None:
        """Start the next backward from dL/dM and dL/dP (packed, fp32) at the segment's end."""
        if self.grad_scaling:
            raise NotImplementedError("cute_ldm: grad_scaling with segments (a carried dM)")
        self.dm.copy_(dm.view_as(self.dm))
        self.covariance_grad.copy_(covariance_grad.view_as(self.covariance_grad))
        self.grad_loaded = True

    @torch.no_grad()
    def export_grad(self, fold: bool = True):
        """dL/dM and dL/dP at the segment's start, after the backward of its first edge: dM with the
        first window's own terms folded in, sum_e q_e dy_e^T - kbar_e dv_e^T over its edges (what
        the previous window's window-end would add from its future tapes)."""
        if self.grad_scaling:
            raise NotImplementedError("cute_ldm: grad_scaling with segments (a carried dM)")
        if not fold:  # (tests: the fold is needed)
            return self.dm.float(), self.covariance_grad
        count = min(self.span, self.total_edges)
        dy = self.dy[:count].float()  # [E, n, g, d] the window's read gradients
        dv = self.dv[:count].float()  # [E, n, g, d] its pseudo-value gradients
        q = self.queries[:count]  # [E, n, g, k]
        kbar = self.keys[:count]
        fold = torch.einsum("engk,engd->ngkd", q, dy) - torch.einsum("engk,engd->ngkd", kbar, dv)
        dm = self.dm.float() + fold
        return dm, self.covariance_grad

    def grad_buffers(self):
        """The buffers the backward starts from (load_grad's targets): dM (storage type), dP fp32."""
        return self.dm, self.covariance_grad

    def state_layout(self):
        """(per-token shape, dtype) of the carried state (M, P as the workspace keeps them) and of
        its gradient (dM in the storage type as the backward keeps it, dP the fp32 packed triangle)."""
        g, k, d = self.groups, self.key_dim, self.padded_value_dim
        dtype = RUNNING_COVARIANCE_DTYPES[self.running_covariance_dtype]
        per16 = 16 // dtype.itemsize
        tri = k * (k + 1) // 2
        state = [((g, k, d), self.storage), ((g, -(-tri // per16) * per16), dtype)]
        grad = [((g, k, d), self.storage), ((g, tri), torch.float32)]
        return state, grad




class CuteRecomputeWorkspace(CuteDepthMemoryWorkspace):
    """The kernels under full activation recompute (Megatron's ``depth_memory_unwind`` with
    ``depth_memory_impl: cute``). Nothing of the state is checkpointed: the per-edge tapes travel
    in the residual stream, the way mHC's streams do.

    Stream. ``[x, records]``, slot first ([slots, ..., h]: x and every record a contiguous
    block): one record per edge of a recompute group, the edge's u in a slot of its own and its
    keys, queries, gains, denominator, forgetting and scale packed bit for bit into shared slots
    (``stream_slots(dtype)``). Megatron checkpoints, and offloads, every
    group's input stream like any residual, so a group's input holds the records of the group
    before it. On the GPU the workspace keeps only the windows of tapes the kernels work on.

    Forward: edges 0 .. L-1 in order, the kernels and their windows of ``span`` unchanged (the
    forward is the one without recompute, bit for bit); ``put_record(edge, records)`` puts the
    edge's record into the stream. Backward: the final read (``ingest`` at the collapse)
    takes the last group's records from the final stream; Megatron then recomputes the groups
    last first, and the first read of a recomputed group takes the previous group's records
    from the group's input stream; the recompute's write passes the records through, and its
    read comes from the running state, which is at a window boundary (the read kernel).
    ``backward(edge)`` runs the window-end and interior launches as without recompute: M by
    subtraction, P by addition from the FP32 running covariance.

    Backward windows. The backward needs only the final state and the records, so its windows
    need not be the forward's. The backward of edge e runs in the backward of the site that
    reads e, the one writing e + 1 (the collapse for the last edge); so a window [s, e)'s
    window-end launch runs once the group of edge e was recomputed, which brings the records of
    the group before it. Every edge of the window must lie in that group or later
    (``layout``): windows of about one group's size (2 edges at 2 layers per group, 3 when they
    end inside a group).

    Records. A record holds the edge's u bit for bit: FP16 u in a BF16 stream is stored as its
    bits (an FP32 stream holds a 16-bit u exactly), so the backward unwinds with the u the forward
    committed into M.

    Checkpoints (``checkpoint_interval`` > 0). The unwinding subtracts each window's writes from a
    later, larger M, so its rounding error follows the largest M of the chain: a burst of large
    writes deep in the stack degrades every state before it. Every ``checkpoint_interval`` edges
    (a multiple of the span, at a boundary b that is a backward window start) the forward copies
    its committed M_b (the state after edges < b) to pinned host memory (``checkpoint_offload``,
    on a side stream) or to the device; the backward replaces its unwound M_b by that copy right
    after the window-end launch that reaches b. The unwinding then never runs across more than
    one interval. P is not checkpointed: its FP32 unwinding is accurate to ~1e-6.
    """

    def __init__(self, *, checkpoint_interval: int = 0, checkpoint_offload: bool = True,
                 checkpoint_prefetch: bool = False, **settings):
        super().__init__(**settings)
        if checkpoint_interval < 0:
            raise ValueError("cute_ldm recompute: checkpoint_interval must be >= 0")
        self.checkpoint_interval = int(checkpoint_interval)
        self.checkpoint_offload = bool(checkpoint_offload)
        # offloaded checkpoints: copy the next one the backward needs into a device staging buffer
        # (one checkpoint's size) an interval ahead, so the restore is a device copy
        self.checkpoint_prefetch = bool(checkpoint_prefetch) and self.checkpoint_offload
        self.windows = None

    # -- layout --------------------------------------------------------------------------------
    def layout(self, groups, width: int) -> None:
        """``groups``: the (first edge, edge count) of every recompute group in order, edge 0
        (the embedding write) a group of its own; ``width``: the stream's width h."""
        groups = [(first, count) for first, count in groups if count > 0]
        edges = [e for first, count in groups for e in range(first, first + count)]
        if edges != list(range(self.total_edges)) or groups[0] != (0, 1):
            raise ValueError(f"cute_ldm recompute: groups {groups} do not cover edges "
                             f"0 .. {self.total_edges - 1} in order with edge 0 alone")
        if width != self.groups * self.value_dim:
            raise ValueError(f"cute_ldm recompute: stream width {width} is not "
                             f"{self.groups} x {self.value_dim}")
        L = self.total_edges
        self.group_ranges = [(first, first + count) for first, count in groups]
        self.group_of = [g for g, (first, end) in enumerate(self.group_ranges) for _ in range(first, end)]
        self.forward_windows = [(s, min(s + self.span, L)) for s in range(0, L, self.span)]
        windows, s = [], 0
        while s < L:
            e = min(s + self.span, L)
            while s < self.group_ranges[self._group_after(e) - 1][0]:
                e -= 1
            windows.append((s, e))
            s = e
        self.windows = windows
        self.window_of = [w for w, (s, e) in enumerate(windows) for _ in range(s, e)]
        starts = {s for s, _ in windows}
        self.checkpoints = []
        if self.checkpoint_interval:
            if self.checkpoint_interval % self.span:
                raise ValueError(f"cute_ldm recompute: checkpoint_interval {self.checkpoint_interval} "
                                 f"is not a multiple of the span {self.span}")
            for boundary in range(self.checkpoint_interval, L, self.checkpoint_interval):
                if boundary not in starts:
                    raise ValueError(f"cute_ldm recompute: checkpoint boundary {boundary} is not a "
                                     f"backward window start (windows {windows})")
                self.checkpoints.append(boundary)
        self.checkpoint_index = {b: i for i, b in enumerate(self.checkpoints)}
        self.width = width
        self.record_slots = max(count for _, count in groups)
        g, k = self.groups, self.key_dim
        self.small_parts = [g * k, g * k, g * k, g, g, g]   # keys, queries, gains, den, forget, scale
        self.small_words = sum(self.small_parts)            # FP32 words per edge
        self.pool = self._windows_alive()
        self.n = None

    def _group_after(self, edge) -> int:
        """The group whose recompute precedes the backward of edge - 1: that of ``edge``, the
        collapse (one past the last group) for the last edge."""
        return len(self.group_ranges) if edge == self.total_edges else self.group_of[edge]

    def _windows_alive(self) -> int:
        """The window slots the GPU needs: the most windows alive at once over the backward
        (a group's predecessor's records arriving at its recompute, the last group's at the
        collapse; a window done once its edges and the window-end launch of the window before it
        are). Also checks that every window-end launch finds its window and the next one."""
        L, groups = self.total_edges, self.group_ranges
        present = set(range(*groups[-1]))
        alive = {self.window_of[e] for e in present}
        most = len(alive)

        def backward(edge):
            w = self.window_of[edge]
            start, end = self.windows[w]
            needed = set(range(start, end))
            if edge == end - 1 and w + 1 < len(self.windows):
                needed |= set(range(*self.windows[w + 1]))
            if not needed <= present:
                raise ValueError(f"cute_ldm recompute: windows {self.windows} do not fit the "
                                 f"recompute groups {groups}")
            for v in [v for v in alive if self.windows[v][0] > edge]:
                alive.discard(v)
                present.difference_update(range(*self.windows[v]))

        backward(L - 1)                       # the final read's backward, before any recompute
        for index in range(len(groups) - 1, 0, -1):
            first, end = groups[index]
            present.update(range(*groups[index - 1]))
            alive.update(self.window_of[e] for e in range(*groups[index - 1]))
            most = max(most, len(alive))
            for edge in range(end - 2, first - 2, -1):
                if edge < L - 1:
                    backward(edge)
        return most

    def stream_slots(self, dtype) -> int:
        """The width of the stream in slots of h: x, the u of every edge of a group, and the
        packed small tapes."""
        return 1 + self.record_slots + self._small_slots(dtype)

    def _small_width(self, dtype) -> int:
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"cute_ldm recompute: a {dtype} stream (bfloat16, float16 or float32)")
        return self.small_words * 4 // dtype.itemsize

    def _small_layout(self, dtype):
        """(records per slot, slots per record) of the small tapes: side by side in a slot when
        they fit, else each over whole slots of their own."""
        small = self._small_width(dtype)
        if small <= self.width:
            return self.width // small, 1
        return 1, -(-small // self.width)

    def _small_slots(self, dtype) -> int:
        per_slot, span = self._small_layout(dtype)
        return -(-self.record_slots // per_slot) * span

    def _put_small(self, records, j, packed) -> None:
        """Record j's small tapes ([n, small] in the stream's dtype) into ``records``."""
        per_slot, span = self._small_layout(records.dtype)
        small, h = packed.shape[1], self.width
        if span == 1:
            start = j % per_slot * small
            records[self.record_slots + j // per_slot][..., start:start + small].copy_(
                packed.view(*records.shape[1:-1], small))
            return
        rows = records[self.record_slots + j * span:self.record_slots + (j + 1) * span]
        padded = torch.nn.functional.pad(packed, (0, span * h - small))
        rows.view(span, -1, h).copy_(padded.view(-1, span, h).transpose(0, 1))

    def _get_small(self, records, j) -> torch.Tensor:
        """Record j's small tapes out of ``records``, as FP32 words [n, small_words]."""
        per_slot, span = self._small_layout(records.dtype)
        small, h, n = self._small_width(records.dtype), self.width, self.n
        if span == 1:
            start = j % per_slot * small
            block = records[self.record_slots + j // per_slot][..., start:start + small].reshape(n, small)
        else:
            rows = records[self.record_slots + j * span:self.record_slots + (j + 1) * span]
            block = rows.reshape(span, n, h).transpose(0, 1).reshape(n, span * h)[:, :small]
        return block.contiguous().view(torch.float32)

    # -- buffers -------------------------------------------------------------------------------
    def ensure(self, n: int, device) -> None:
        if self.windows is None:
            raise RuntimeError("cute_ldm recompute: layout() before the first step")
        if self.n == n and self.m.device == torch.device(device):
            return
        super().ensure(n, device)
        self.slot_of = {}          # window -> its slot of the window tapes
        self._present = set()      # edges whose tapes are in a slot
        self._phase, self._next, self._boundary = "forward", 0, 0
        self._state_edge = 0       # the window boundary the running state M is at (edges before it applied)
        self._read_cache = {}      # (window, from_end, state) -> reads of the window [E, n, g, d]
        self._pending = None       # (edge, read) of the forward's last step, for the next site
        self._written = None       # (edge, slot, row) the forward just wrote, for put_record
        self._final_consumed = False   # the last edge's read, taken once after the forward
        self.grad_stash = {}       # edge -> the step's input gradients, from the read's backward
        self._allocate_checkpoints(n)

    def _allocate_checkpoints(self, n: int) -> None:
        """The M copies at the checkpoint boundaries, [checkpoint, n, G, K, D] in the storage type."""
        count = len(self.checkpoints)
        self.checkpoint_m, self._checkpoint_pending, self._staged = None, None, None
        if not count:
            return
        shape = (count, n, self.groups, self.key_dim, self.padded_value_dim)
        if self.checkpoint_offload:
            self.checkpoint_m = torch.empty(shape, dtype=self.storage, pin_memory=True)
        else:
            self.checkpoint_m = torch.empty(shape, dtype=self.storage, device=self.device)
        self._checkpoint_stream = torch.cuda.Stream(device=self.device)
        self._checkpoint_done = [torch.cuda.Event() for _ in range(count)]
        self._staged, self._staged_done = None, torch.cuda.Event()
        if self.checkpoint_prefetch:
            self.checkpoint_staging = torch.empty(shape[1:], dtype=self.storage, device=self.device)

    def _prefetch_checkpoint(self, index: int) -> None:
        """Copy checkpoint ``index`` into the staging buffer on the side stream, after the device
        copy out of it that the main stream has queued (the previous restore)."""
        side = self._checkpoint_stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.checkpoint_staging.copy_(self.checkpoint_m[index], non_blocking=True)
            self._staged_done.record(side)
        self._staged = index

    def _save_checkpoint(self, index: int) -> None:
        """Copy the committed M (the state at boundary ``checkpoints[index]``) on the side stream;
        the next launch that writes M waits for it (_wait_checkpoint)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cute_ldm recompute: M checkpoints inside a CUDA graph capture")
        side = self._checkpoint_stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.checkpoint_m[index].copy_(self.m, non_blocking=True)
            self._checkpoint_done[index].record(side)
        self._checkpoint_pending = index

    def _wait_checkpoint(self) -> None:
        """Before a launch that writes M: the pending copy of M has finished."""
        if self._checkpoint_pending is not None:
            torch.cuda.current_stream().wait_event(self._checkpoint_done[self._checkpoint_pending])
            self._checkpoint_pending = None

    def _restore_checkpoint(self, boundary: int) -> None:
        """The backward reached ``boundary``: its unwound M becomes the forward's copy."""
        index = self.checkpoint_index[boundary]
        stream = torch.cuda.current_stream()
        if self._staged == index:
            stream.wait_event(self._staged_done)
            self.m.copy_(self.checkpoint_staging)
        else:
            stream.wait_event(self._checkpoint_done[index])
            self.m.copy_(self.checkpoint_m[index], non_blocking=True)
        self._staged = None
        if self.checkpoint_prefetch and index > 0:
            self._prefetch_checkpoint(index - 1)

    # -- u between the storage type and the stream's ------------------------------------------
    def _u_to_record(self, u, dtype):
        """u (storage type) as the stream's ``dtype``, without rounding: the bits when the widths
        match (FP16 in a BF16 stream), else the value (exact in an FP32 stream)."""
        if u.dtype == dtype:
            return u
        if u.dtype.itemsize == dtype.itemsize:
            return u.contiguous().view(dtype)
        return u.to(dtype)

    def _u_from_record(self, u):
        """The inverse of _u_to_record: a record's u in the storage type."""
        if u.dtype == self.storage:
            return u
        if u.dtype.itemsize == self.storage.itemsize:
            return u.contiguous().view(self.storage)
        return u.to(self.storage)

    def _backward_rows(self) -> int:
        return max(end - start for start, end in self.windows)

    def _allocate_reads(self, n: int) -> None:
        pass   # the reads of recomputed layers are allocated per window (_recomputed_read)

    def _allocate_tapes(self, n: int) -> None:
        """The per-edge tapes: ``pool`` backward windows of [slot][local edge] rows, and in the
        same memory (never live at once) the forward's window of ``span`` rows."""
        g, k, d, P, E = self.groups, self.key_dim, self.padded_value_dim, self.pool, self._backward_rows()
        rows = max(P * E, self.span)

        def tape(*shape, dtype=torch.float32):
            flat = torch.empty((rows, n, g, *shape), dtype=dtype, device=self.device)
            return flat[:P * E].view(P, E, n, g, *shape), flat[:self.span].view(1, self.span, n, g, *shape)

        names = ("keys", "queries", "gains", "residuals", "denominators", "forgetting_tape", "scales")
        shapes = ((k,), (k,), (k,), (d,), (), (), ())
        self._forward_tapes = {}
        for name, shape in zip(names, shapes):
            pool, forward = tape(*shape, dtype=self.storage if name == "residuals" else torch.float32)
            setattr(self, name, pool)
            self._forward_tapes[name] = forward

    def _slot(self, window):
        """The slot of a window whose tapes are on the GPU."""
        if window not in self.slot_of:
            raise RuntimeError(f"cute_ldm recompute: the tapes of window {self.windows[window]} "
                               "are not in the workspace")
        return self.slot_of[window]

    def _place(self, window):
        """The slot for ``window``'s tapes: a free one, or that of a window whose edges and the
        window-end launch of the window before it are done."""
        if window in self.slot_of:
            return self.slot_of[window]
        used = set(self.slot_of.values())
        free = [slot for slot in range(self.pool) if slot not in used]
        if not free:
            done = [w for w in self.slot_of if self._boundary < self.windows[w][0]]
            if not done:
                raise RuntimeError(f"cute_ldm recompute: no slot for window "
                                   f"{self.windows[window]} (pool {self.pool})")
            victim = max(done)
            free = [self.slot_of.pop(victim)]
            start, end = self.windows[victim]
            self._present.difference_update(range(start, end))
            self._drop_reads(victim)
        self.slot_of[window] = free[0]
        return free[0]

    # -- the records in the stream -------------------------------------------------------------
    @torch.no_grad()
    def put_record(self, edge, records) -> None:
        """Write the record of ``edge`` into ``records`` (the stream past x, [slots - 1, ..., h])
        if the forward just stepped it; under recompute the records pass through as they are."""
        if self._written is None or self._written[0] != edge:
            return
        _, slot, local = self._written
        self._written = None
        n, t = self.n, self._forward_tapes
        j = edge - self.group_ranges[self.group_of[edge]][0]
        u = self._unpadded(t["residuals"][slot, local]).reshape(records.shape[1:])
        records[j].copy_(self._u_to_record(u, records.dtype))
        packed = torch.cat([t["keys"][slot, local].reshape(n, -1),
                            t["queries"][slot, local].reshape(n, -1),
                            t["gains"][slot, local].reshape(n, -1),
                            t["denominators"][slot, local], t["forgetting_tape"][slot, local],
                            t["scales"][slot, local]], dim=1).view(records.dtype)
        self._put_small(records, j, packed)

    @torch.no_grad()
    def ingest(self, edge, streams) -> None:
        """Under recompute, at a read of the group of ``edge``: copy the records of the group
        before it from the group's input ``streams`` into the workspace (once)."""
        if self._phase != "backward":
            return
        index = self._group_after(edge)
        if index == 0:
            return
        first, end = self.group_ranges[index - 1]
        missing = [e for e in range(first, end) if e not in self._present]
        if not missing:
            return
        n, g, k = self.n, self.groups, self.key_dim
        records = streams.detach()[1:]
        for e in missing:
            slot = self._place(self.window_of[e])
            local = e - self.windows[self.window_of[e]][0]
            j = e - first
            u = self._u_from_record(records[j].reshape(n, g, self.value_dim))
            self.residuals[slot, local].copy_(self._padded(u))
            words = self._get_small(records, j)
            keys, queries, gains, den, forget, scale = words.split(self.small_parts, dim=1)
            self.keys[slot, local].copy_(keys.reshape(n, g, k))
            self.queries[slot, local].copy_(queries.reshape(n, g, k))
            self.gains[slot, local].copy_(gains.reshape(n, g, k))
            self.denominators[slot, local].copy_(den)
            self.forgetting_tape[slot, local].copy_(forget)
            self.scales[slot, local].copy_(scale)
            self._present.add(e)

    # -- forward -------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, edge, khat, query, value, scale, forgetting) -> torch.Tensor:
        n, g, k = self.n, self.groups, self.key_dim
        if edge == 0 or self._phase == "forward":
            if edge == 0:
                self._phase, self._next = "forward", 0
                self._final_consumed = False
                self.slot_of.clear()
                self._present.clear()
                self._read_cache.clear()
                self.grad_stash.clear()
                self.p.copy_(self.p0)
            if edge != self._next:
                raise RuntimeError(f"cute_ldm recompute: forward edge {edge}, expected {self._next}")
            # the forward's window in slot 0: its rows go to the stream right after each step
            start, end = self.forward_windows[edge // self.span]
            slot = 0
            raw_key = _aligned(khat.float().reshape(n, g, k))
            query = _aligned(query.float().reshape(n, g, k))
            scale = _aligned(scale.float().reshape(n, g))
            forgetting = torch.broadcast_to(forgetting.float(), (n, g))
            if forgetting.stride() != (0, 1):
                forgetting = _aligned(forgetting)
            value = self._padded(self._to_storage(value).contiguous().view(n, g, self.value_dim))
            read = torch.empty_like(value)
            local = edge - start
            t = self._forward_tapes
            tapes = [t["keys"][slot, local], t["queries"][slot, local], t["scales"][slot, local],
                     t["forgetting_tape"][slot, local]]
            sizes = (end - start, n, g, k, self.padded_value_dim)
            options = (("first_window", start == 0), ("running_covariance_dtype", self.running_covariance_dtype))
            count = end - start
            if local == count - 1:
                self._wait_checkpoint()   # this launch commits the window into M
            _launch("forward", sizes,
                    [self.m, self.p, raw_key, query, value, scale, forgetting, t["gains"][slot, :count],
                     t["residuals"][slot, :count], t["denominators"][slot, :count], read, *tapes],
                    local, options=options, training=True)
            if edge + 1 in self.checkpoint_index:
                self._save_checkpoint(self.checkpoint_index[edge + 1])
            self._written = (edge, slot, local)
            self._next = edge + 1
            self._pending = (edge, read)
            if self._next == self.total_edges:
                # the backward fills the slots from the records (the final stream's first)
                self._phase, self._boundary = "backward", self.total_edges
                self._state_edge = self.total_edges
                self.slot_of.clear()
                self._present.clear()
            return self._unpadded(read)
        # the recompute of the step's layer: the tapes hold everything; its read comes from read()
        return None

    def read(self, edge) -> torch.Tensor:
        """The read of edge ``edge`` for the sublayer after it, [n, G, D] (storage type): in the forward the
        step's own read, under recompute the read kernel's from the running state."""
        pending = self._pending is not None and self._pending[0] == edge
        if self._phase == "forward" or (pending and not self._final_consumed):
            if not pending:
                raise RuntimeError(f"cute_ldm recompute: read of edge {edge} before its step")
            if edge == self.total_edges - 1:
                self._final_consumed = True
            return self._unpadded(self._pending[1])
        return self._recomputed_read(edge)

    def _recomputed_read(self, edge):
        w = self.window_of[edge]
        start, end = self.windows[w]
        # The running state sits at a window boundary: at or after this window's end until its
        # window-end launch (its last edge's backward), at its start after it, whatever interior
        # edges remain. A group that straddles windows is recomputed with the state past this
        # window's end; the later windows' writes up to the state are then subtracted as well.
        state = self._state_edge
        if state >= end:
            from_end = True
        elif state == start:
            from_end = False
        else:
            raise RuntimeError(f"cute_ldm recompute: edge {edge} recomputed with the state at edge "
                               f"boundary {state}, not at its window {start}-{end}")
        key = (w, from_end, state)
        if key not in self._read_cache:
            slot = self._slot(w)
            n, g, k, d = self.n, self.groups, self.key_dim, self.padded_value_dim
            count = end - start
            reads = torch.empty((count, n, g, d), dtype=self.storage, device=self.device)
            _launch("read", (count, n, g, k, d),
                    [self.m, self.queries[slot, :count], self.gains[slot, :count],
                     self.residuals[slot, :count], reads],
                    options=(("lo", 0), ("hi", count), ("from_end", from_end)), training=True)
            later = w + 1
            while from_end and later < len(self.windows) and self.windows[later][0] < state:
                lstart, lend = self.windows[later]
                lslot = self._slot(later)
                used = min(lend, state) - lstart
                # r_j -= sum_i (kappa_i . q_j) u_i over the later window's edges before the state
                coef = torch.einsum("ingk,jngk->ijng", self.gains[lslot, :used], self.queries[slot, :count])
                reads -= torch.einsum("ijng,ingd->jngd", coef, self.residuals[lslot, :used].float()).to(reads.dtype)
                later += 1
            self._read_cache[key] = reads
        return self._unpadded(self._read_cache[key][edge - start])

    # -- backward ------------------------------------------------------------------------------
    @torch.no_grad()
    def backward(self, edge, read_grad):
        if self._phase != "backward" or edge != self._boundary - 1:
            raise RuntimeError(f"cute_ldm recompute: backward of edge {edge} out of order "
                               f"(the state is at boundary {self._boundary})")
        w = self.window_of[edge]
        start, end = self.windows[w]
        slot = self._slot(w)
        if edge == self.total_edges - 1 and not self.grad_loaded:
            self.dm.zero_()
            self.covariance_grad.zero_()
            self._begin_backward(read_grad)
            if self.checkpoint_prefetch and self.checkpoints:
                self._prefetch_checkpoint(len(self.checkpoints) - 1)
        self._wait_checkpoint()   # the window-end launches write M
        zero_start = start == 0 and not self.state_loaded
        count = end - start
        scratch = self.p_scratch[:count]
        n, g, k, d = self.n, self.groups, self.key_dim, self.padded_value_dim
        query_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        forgetting_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        query_output = query_grad.expand(count, n, g, k)
        forgetting_output = forgetting_grad.expand(count, n, g)
        raw_value_grad = torch.empty((n, g, d), device=self.device, dtype=self.storage)
        raw_key_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        scale_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        current_dy = _aligned(self._padded(self._cotangent(read_grad).reshape(n, g, self.value_dim)))
        if edge == end - 1:
            if w + 1 < len(self.windows):
                fslot = self._slot(w + 1)
                future = self.windows[w + 1][1] - self.windows[w + 1][0]
                fq, fk = self.queries[fslot, :future], self.keys[fslot, :future]
            else:
                future = 0
                fq, fk = self.queries[slot, :1], self.keys[slot, :1]   # F=0 placeholders never read
            _launch("window_end", (count, future, n, g, k, d),
                    [self.m, self.dm, self.keys[slot, :count], self.queries[slot, :count],
                     self.gains[slot, :count], self.residuals[slot, :count],
                     self.forgetting_tape[slot, :count], self.denominators[slot, :count],
                     fq, fk, self.dy[:max(1, future)], self.dv[:max(1, future)], self.m, self.dm,
                     self.future_value_grad[:count], self.future_gain_grad[:count], current_dy,
                     self.scales[slot, edge - start], self.dy[:count], self.dv[:count], query_output,
                     self.covariance_grad, forgetting_output,
                     raw_value_grad, raw_key_grad, scale_grad, self.p, scratch],
                    options=(("first_window", zero_start),), training=True)
            self._state_edge = start
            self._drop_reads(w, from_end=True)   # the state has left the window's end
            if start in self.checkpoint_index:
                self._restore_checkpoint(start)
        else:
            _launch("interior", (count, n, g, k, d),
                    [self.m, self.residuals[slot, :count], self.dy[:count], self.keys[slot, :count],
                     self.gains[slot, :count], self.queries[slot, :count],
                     self.future_value_grad[:count], self.future_gain_grad[:count],
                     self.dv[:count], query_output, self.covariance_grad,
                     self.forgetting_tape[slot, :count], self.denominators[slot, :count],
                     forgetting_output, self.scales[slot, edge - start], raw_value_grad, raw_key_grad,
                     scale_grad, scratch, current_dy], edge - start, training=True)
        self._boundary = edge
        if edge == start:
            self._drop_reads(w)      # the window's backward is done
        if edge == 0:
            self._phase, self._next = "forward", 0
            self._end_backward()
        return self._input_grads(raw_key_grad, query_grad, self._unpadded(raw_value_grad),
                                 scale_grad, forgetting_grad.sum(0))

    def _drop_reads(self, window, from_end=None):
        for key in [key for key in self._read_cache if key[0] == window
                    and (from_end is None or key[1] == from_end)]:
            del self._read_cache[key]
