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

    types = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    fake = [None if item is None else make_fake_tensor(
        types[item[0]], item[1], item[2], assumed_align=item[3]) for item in metadata]
    if kernel != "window_end":
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


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` if it is contiguous and 16-byte aligned, else a contiguous copy."""
    if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


class CuteDepthMemoryWorkspace:
    """The tapes and buffers of ``total_edges`` LDM steps for ``n`` tokens (see module doc)."""

    def __init__(self, *, total_edges: int, groups: int, key_dim: int, value_dim: int,
                 alpha: float = 1.0, span: int = 4, running_covariance_dtype: str = "float32"):
        if key_dim not in (4, 8, 16):
            raise ValueError("cute_ldm: the key dimension must be 4, 8 or 16")
        if running_covariance_dtype not in RUNNING_COVARIANCE_DTYPES:
            raise ValueError(f"running covariance dtype must be one of "
                             f"{sorted(RUNNING_COVARIANCE_DTYPES)}")
        self.total_edges, self.span = total_edges, min(span, total_edges)
        self.groups, self.key_dim, self.value_dim = groups, key_dim, value_dim
        self.padded_value_dim = -(-value_dim // 8) * 8
        self.covariance_scale = float(alpha)
        self.running_covariance_dtype = running_covariance_dtype
        self.n = None

    # -- buffers -------------------------------------------------------------------------------
    def ensure(self, n: int, device) -> None:
        if self.n == n and self.m.device == torch.device(device):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cute_ldm: workspace allocation inside a CUDA graph capture")
        self.n, self.device = n, torch.device(device)
        g, k, d, length, span = self.groups, self.key_dim, self.padded_value_dim, \
            self.total_edges, self.span

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        self.m = empty((n, g, k, d), torch.bfloat16)
        dtype = RUNNING_COVARIANCE_DTYPES[self.running_covariance_dtype]
        per_16_bytes = 16 // dtype.itemsize
        triangle = k * (k + 1) // 2
        packed = torch.zeros((-(-triangle // per_16_bytes) * per_16_bytes,), dtype=dtype,
                             device=self.device)
        row = torch.arange(k, device=self.device)
        packed.index_fill_(0, row * k - row * (row - 1) // 2, self.covariance_scale)
        self.p0 = packed.expand(n, g, packed.numel()).clone()
        self.p = self.p0.clone()
        self.keys, self.queries, self.gains = [empty((length, n, g, k)) for _ in range(3)]
        self.residuals = empty((length, n, g, d), torch.bfloat16)
        self.denominators, self.forgetting_tape, self.scales = [empty((length, n, g))
                                                                for _ in range(3)]
        self.saved_covariance_tape = empty((length, n, g, triangle), torch.bfloat16)
        self.dm = empty((n, g, k, d), torch.bfloat16)
        self.covariance_grad = empty((n, g, triangle))
        self.future_value_grad = empty((span, n, g, d))
        self.dy, self.dv = [empty((span, n, g, d), torch.bfloat16) for _ in range(2)]
        self.future_gain_grad = empty((span, n, g, k))
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

    # -- forward -------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, edge, khat, query, value, scale, forgetting) -> torch.Tensor:
        n, g, k = self.n, self.groups, self.key_dim
        if edge == 0:
            self.p.copy_(self.p0)
        start, end = self._window(edge)
        raw_key = _aligned(khat.float().reshape(n, g, k))
        query = _aligned(query.float().reshape(n, g, k))
        scale = _aligned(scale.float().reshape(n, g))
        forgetting = torch.broadcast_to(forgetting.float(), (n, g))
        if forgetting.stride() != (0, 1):
            forgetting = _aligned(forgetting)
        value = self._padded(value.to(torch.bfloat16).contiguous().view(n, g, self.value_dim))
        read = torch.empty_like(value)
        tapes = [self.keys[edge], self.queries[edge], self.scales[edge], self.forgetting_tape[edge]]
        sizes = (end - start, n, g, k, self.padded_value_dim)
        options = (("first_window", start == 0),
                   ("running_covariance_dtype", self.running_covariance_dtype))
        _launch("forward", sizes,
                [self.m, self.p, raw_key, query, value, scale, forgetting, self.gains[start:end],
                 self.residuals[start:end], self.denominators[start:end], read,
                 self.saved_covariance_tape[start:end], *tapes],
                edge - start, options=options, training=True)
        return self._unpadded(read)

    # -- backward ------------------------------------------------------------------------------
    @torch.no_grad()
    def backward(self, edge, read_grad):
        if edge == self.total_edges - 1:
            self.dm.zero_()
            self.covariance_grad.zero_()
        start, end = self._window(edge)
        count = end - start
        saved = self.saved_covariance_tape[start:end]
        n, g, k, d = self.n, self.groups, self.key_dim, self.padded_value_dim
        query_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        forgetting_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        query_output = query_grad.expand(count, n, g, k)
        forgetting_output = forgetting_grad.expand(count, n, g)
        raw_value_grad = torch.empty((n, g, d), device=self.device, dtype=torch.bfloat16)
        raw_key_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        scale_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        current_dy = _aligned(self._padded(
            read_grad.to(torch.bfloat16).reshape(n, g, self.value_dim)))
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
                     raw_value_grad, raw_key_grad, scale_grad, saved],
                    options=(("first_window", start == 0),), training=True)
        else:
            _launch("interior", (count, n, g, k, d),
                    [self.m, self.residuals[start:end], self.dy[:count], self.keys[start:end],
                     self.gains[start:end], self.queries[start:end],
                     self.future_value_grad[:count], self.future_gain_grad[:count],
                     self.dv[:count], query_output, self.covariance_grad,
                     self.forgetting_tape[start:end], self.denominators[start:end],
                     forgetting_output, self.scales[edge], raw_value_grad, raw_key_grad,
                     scale_grad, saved, current_dy], edge - start, training=True)
        return (raw_key_grad, query_grad, self._unpadded(raw_value_grad), scale_grad,
                forgetting_grad.sum(0))
