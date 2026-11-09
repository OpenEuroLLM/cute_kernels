"""The low-bit expert kernels against the fp32 reference of the same quantized weights: every format, token
counts from 1 to 64, single and grouped launches; the int4 packing round-trips."""

import pytest
import torch

from lowbit_experts import expert_ffn, expert_ffn_reference, quantize, stack
from lowbit_experts.reference import dequantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
H, I = 512, 256  # small multiples of the group sizes (the v4 shape is 2048 / 768)


def _weights(fmt, group, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wgu = torch.randn(2 * I, H, device="cuda", generator=g) * 0.02
    wd = torch.randn(H, I, device="cuda", generator=g) * 0.02
    return quantize(wgu, fmt, group), quantize(wd, fmt, group)


@pytest.mark.parametrize("fmt,group", [("bf16", 128), ("fp8", 128), ("int8", 128), ("int4", 128), ("int4", 64)])
@pytest.mark.parametrize("m", [1, 7, 16, 64])
def test_matches_reference(fmt, group, m):
    a, b = _weights(fmt, group)
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
    y = expert_ffn(x, a, b).float()
    ref = expert_ffn_reference(x, a, b)
    assert float((y - ref).abs().max() / ref.abs().max()) < 2e-2  # bf16 activations / intermediate


@pytest.mark.parametrize("fmt", ["fp8", "int8", "int4"])
def test_grouped_equals_single(fmt):
    ws = [_weights(fmt, 128, seed=s) for s in range(3)]
    a, b = stack([w[0] for w in ws]), stack([w[1] for w in ws])
    x = torch.randn(3, 5, H, device="cuda", dtype=torch.bfloat16)
    y = expert_ffn(x, a, b)
    for e in range(3):
        torch.testing.assert_close(y[e], expert_ffn(x[e], *ws[e]), atol=0, rtol=0)


def test_int4_roundtrip():
    w = torch.randn(64, 256, device="cuda")
    q = quantize(w, "int4", 64)
    d = dequantize(q)
    s = q.scale.repeat_interleave(64, 1)
    assert bool(((d - w).abs() <= s * 0.5 + 1e-6).all())  # within half a step of the full-range RTN grid
