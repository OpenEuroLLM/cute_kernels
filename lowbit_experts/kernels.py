"""Triton kernels: y = a W^T with W stored low-bit and dequantized in registers inside the GEMM.

Formats (``LowBitWeight.fmt``):
  bf16  plain bf16 weights (the baseline through the same kernel)
  fp8   E4M3 bytes + one fp32 scale per ``group`` (128) consecutive elements of a row (the oellm_ambitious export:
        Megatron's fp8_param_gather layout); decoded in software (sm_86 has no FP8 conversion): the 7 magnitude
        bits shifted into an fp16 bit pattern give the value * 2^-8 exactly, subnormals included
  int8  int8 + one fp32 scale per output row, applied to the accumulator
  int4  symmetric int4 (stored + 8, two per byte: element 2j in the low nibble) + one fp32 scale per ``group``
        (64 / 128) elements of a row

``expert_ffn`` runs a SwiGLU expert in two launches: gate + up as one GEMM (``[M, 2I]``), then down with
silu(gate) * up computed while its input tiles load.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

FORMATS = {"bf16": 0, "fp8": 1, "int8": 2, "int4": 3}


@dataclass
class LowBitWeight:
    """A [n, k] weight in one of FORMATS: ``data`` (bf16 [n, k] / uint8 [n, k] E4M3 bytes / int8 [n, k] /
    uint8 [n, k / 2] packed int4), ``scale`` (fp32: [n, k / group] for fp8 / int4, [n] for int8, None for bf16)."""

    fmt: str
    data: torch.Tensor
    scale: torch.Tensor | None
    n: int
    k: int
    group: int = 128

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size() + (
            0 if self.scale is None else self.scale.numel() * self.scale.element_size())


def _search_scale(w: torch.Tensor, qmax: float, ratios) -> torch.Tensor:
    """Per-group scale = r * amax / qmax with the r of the lowest squared error (w: [G, group])."""
    amax = w.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    best, best_err = None, None
    for r in ratios:
        s = r * amax / qmax
        q = (w / s).round().clamp(-qmax - 1, qmax)
        err = ((q * s - w) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = s, err
        else:
            better = err < best_err
            best = torch.where(better, s, best)
            best_err = torch.where(better, err, best_err)
    return best


def quantize(w: torch.Tensor, fmt: str, group: int = 128, search: bool = False) -> LowBitWeight:
    """Quantize a [n, k] float weight. ``search``: per-group clipping ratio by least squared error (int4 / int8)
    instead of round-to-nearest at the full range."""
    n, k = w.shape
    w = w.float()
    if fmt == "bf16":
        return LowBitWeight("bf16", w.to(torch.bfloat16).contiguous(), None, n, k, group)
    if fmt == "fp8":
        g = w.view(n, k // group, group)
        s = (g.abs().amax(-1) / 448.0).clamp(min=1e-12)
        q = (g / s[..., None]).to(torch.float8_e4m3fn).view(torch.uint8).view(n, k)
        return LowBitWeight("fp8", q.contiguous(), s.contiguous(), n, k, group)
    ratios = torch.linspace(0.5, 1.0, 11).tolist() if search else [1.0]
    if fmt == "int8":
        s = _search_scale(w, 127.0, ratios)[:, 0]
        q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
        return LowBitWeight("int8", q.contiguous(), s.contiguous(), n, k, group)
    if fmt == "int4":
        g = w.view(n * (k // group), group)
        s = _search_scale(g, 7.0, ratios)
        q = ((g / s).round().clamp(-8, 7) + 8).to(torch.int16).view(n, k)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
        return LowBitWeight("int4", packed.contiguous(), s.view(n, k // group).contiguous(), n, k, group)
    raise ValueError(fmt)


@triton.jit
def _decode_e4m3(b):
    v = b.to(tl.int16)
    bits = ((v & 0x80) << 8) | ((v & 0x7F) << 7)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


def _configs():
    return [triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
            for bn in (16, 32, 64) for bk in (64, 128) for w in (2, 4) for st in (2, 3, 4)]


def _prune(configs, named_args, **kwargs):
    """fp8 / int4: one scale per row per K tile (BK divides the group)."""
    group = kwargs.get("GROUP", named_args.get("GROUP"))
    fmt = kwargs.get("FMT", named_args.get("FMT"))
    out = [c for c in configs if fmt not in (1, 3) or group % c.kwargs["BK"] == 0]
    return out or configs[:1]


@triton.autotune(configs=_configs(), key=["M_BUCKET", "N", "K", "FMT", "GROUP", "SWIGLU", "E_BUCKET"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _lowbit_gemm_kernel(A, W, S, Y, M, N, K, I, stride_am, stride_wn, stride_sn, stride_ym, M_BUCKET,
                        stride_ae, stride_we, stride_se, stride_ye, E_BUCKET,
                        FMT: tl.constexpr, GROUP: tl.constexpr, SWIGLU: tl.constexpr, BM: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.program_id(2).to(tl.int64)  # the expert (grouped launch: one weight per expert)
    A += e * stride_ae
    W += e * stride_we
    S += e * stride_se
    Y += e * stride_ye
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    mmask = m < M
    nmask = n < N
    kk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        # A tile [BM, BK] (SWIGLU: silu(gate) * up from the [M, 2I] gate/up rows, K = I)
        if SWIGLU:
            g = tl.load(A + m[:, None] * stride_am + (k0 + kk)[None, :], mask=mmask[:, None], other=0.0)
            u = tl.load(A + m[:, None] * stride_am + (I + k0 + kk)[None, :], mask=mmask[:, None], other=0.0)
            gf = g.to(tl.float32)
            a = (gf / (1.0 + tl.exp(-gf)) * u.to(tl.float32)).to(tl.bfloat16)
        else:
            a = tl.load(A + m[:, None] * stride_am + (k0 + kk)[None, :], mask=mmask[:, None], other=0.0)
        # W tile [BN, BK] dequantized to bf16
        if FMT == 0:
            w = tl.load(W + n[:, None] * stride_wn + (k0 + kk)[None, :], mask=nmask[:, None], other=0.0)
        elif FMT == 1:
            wb = tl.load(W + n[:, None] * stride_wn + (k0 + kk)[None, :], mask=nmask[:, None], other=0)
            sc = tl.load(S + n * stride_sn + k0 // GROUP, mask=nmask, other=0.0)  # BK divides GROUP
            w = (_decode_e4m3(wb) * (sc * 256.0)[:, None]).to(tl.bfloat16)
        elif FMT == 2:
            w = tl.load(W + n[:, None] * stride_wn + (k0 + kk)[None, :], mask=nmask[:, None], other=0)
            w = w.to(tl.bfloat16)  # exact; the per-row scale goes on the accumulator
        else:
            kh = tl.arange(0, BK // 2)
            wb = tl.load(W + n[:, None] * stride_wn + (k0 // 2 + kh)[None, :], mask=nmask[:, None], other=0)
            wb = wb.to(tl.int16)
            q = tl.reshape(tl.join((wb & 0xF) - 8, (wb >> 4) - 8), (BN, BK))  # element 2j low, 2j + 1 high
            sc = tl.load(S + n * stride_sn + k0 // GROUP, mask=nmask, other=0.0)  # BK divides GROUP
            w = (q.to(tl.float32) * sc[:, None]).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(w), acc)
    if FMT == 2:
        acc = acc * tl.load(S + n, mask=nmask, other=0.0)[None, :]
    tl.store(Y + m[:, None] * stride_ym + n[None, :], acc.to(Y.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


def lowbit_linear(a: torch.Tensor, w: LowBitWeight, swiglu_in: bool = False, out: torch.Tensor | None = None):
    """a [M, K] bf16 (``swiglu_in``: [M, 2K] gate | up, silu(gate) * up taken as the input) -> [M, N] bf16.
    Grouped: ``a`` [E, M, K] and ``w`` with a leading expert dimension (``stack``) -> [E, M, N], one launch."""
    grouped = a.dim() == 3
    if not grouped:
        a = a[None]
    e_n, m = a.shape[0], a.shape[1]
    k = a.shape[2] // 2 if swiglu_in else a.shape[2]
    assert k == w.k, (k, w.k)
    a = a if a.stride(-1) == 1 else a.contiguous()
    y = out if out is not None else torch.empty((e_n, m, w.n), dtype=torch.bfloat16, device=a.device)
    data = w.data if w.data.dim() == 3 else w.data[None]
    s = w.scale if w.scale is not None else w.data
    s = s if s.dim() == data.dim() - (1 if w.fmt == "int8" else 0) else s[None]
    assert data.shape[0] == e_n, (data.shape, e_n)
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    grid = lambda meta: (triton.cdiv(m, bm), triton.cdiv(w.n, meta["BN"]), e_n)  # noqa: E731
    _lowbit_gemm_kernel[grid](
        a, data, s, y, m, w.n, k, k, a.stride(1), data.stride(1),
        s.stride(1) if s.dim() > 2 else 1, y.stride(1), bm,
        a.stride(0), data.stride(0), s.stride(0), y.stride(0), min(e_n, 64),
        FMT=FORMATS[w.fmt], GROUP=w.group, SWIGLU=swiglu_in, BM=bm,
    )
    return y if grouped else y[0]


def stack(weights: list[LowBitWeight]) -> LowBitWeight:
    """One LowBitWeight with a leading expert dimension (for grouped launches)."""
    w0 = weights[0]
    return LowBitWeight(w0.fmt, torch.stack([w.data for w in weights]),
                        None if w0.scale is None else torch.stack([w.scale for w in weights]), w0.n, w0.k, w0.group)


def expert_ffn(x: torch.Tensor, w_gate_up: LowBitWeight, w_down: LowBitWeight) -> torch.Tensor:
    """A SwiGLU expert, x [M, H] bf16 -> [M, H] bf16 (gate rows first in w_gate_up), in two launches; grouped:
    x [E, M, H] with ``stack``-ed weights -> [E, M, H], still two launches for all E experts."""
    gu = lowbit_linear(x, w_gate_up)
    return lowbit_linear(gu, w_down, swiglu_in=True)
