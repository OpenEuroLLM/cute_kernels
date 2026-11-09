"""Float references: dequantize a LowBitWeight, the expert FFN in fp32."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dequantize(w) -> torch.Tensor:
    """LowBitWeight -> fp32 [N, K] (exactly what the kernels multiply with, before the bf16 rounding of the fp8
    path)."""
    n, k = w.n, w.k
    if w.fmt == "bf16":
        return w.data.float()
    if w.fmt == "fp8":
        q = w.data.view(torch.float8_e4m3fn).float()
        return (q.view(n, k // w.group, w.group) * w.scale.float()[..., None]).view(n, k)
    if w.fmt == "int8":
        return w.data.float() * w.scale.float()[:, None]
    if w.fmt == "int4":
        b = w.data.to(torch.int16)
        lo, hi = (b & 0xF) - 8, (b >> 4) - 8
        q = torch.stack([lo, hi], -1).view(n, k).float()
        return (q.view(n, k // w.group, w.group) * w.scale.float()[..., None]).view(n, k)
    raise ValueError(w.fmt)


def expert_ffn_reference(x: torch.Tensor, w_gate_up, w_down) -> torch.Tensor:
    """x [M, H] -> [M, H] in fp32: silu(x W_g^T) * (x W_u^T) then W_d^T (gate rows first in W_gate_up)."""
    gu = x.float() @ dequantize(w_gate_up).t()
    i = gu.shape[1] // 2
    return (F.silu(gu[:, :i]) * gu[:, i:]) @ dequantize(w_down).t()
