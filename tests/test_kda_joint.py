"""Triadic KDA / Complex KDA: the CuTe pipeline against the same pipeline with its CuTe stages in torch (backend
"torch", itself checked against the token recurrence on the CPU-sized cases below), forward and all gradients.
`pytest tests/test_kda_joint.py`."""
import pytest
import torch

from cute_triadic_gdn.ops.kda_joint import chunk_kda_joint
from cute_triadic_gdn.ops.kda_reference import naive_kda_joint


def _hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def make_inputs(B, T, H, E, signed, seed, gate_scale=0.5):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    n = lambda x: torch.nn.functional.normalize(x, dim=-1)
    q = n(r(B, T, H, 128)).bfloat16()
    k = n(r(B, T, H, 128)).bfloat16()
    v = (0.5 * r(B, T, H, 128)).bfloat16()
    k2 = n(torch.nn.functional.softplus(r(B, T, H, E)))
    q2 = n(torch.nn.functional.softplus(r(B, T, H, E)))
    g2 = -torch.nn.functional.softplus(r(B, T, H, E) + 3.0) * 0.3
    g = -torch.nn.functional.softplus(r(B, T, H, 128) - 1.0) * gate_scale
    beta = torch.rand(B, T, H, device="cuda", generator=gen) * (2.0 if signed else 1.0)
    sign = None
    if signed:
        sign = torch.where(torch.rand(B, T, H, 128, device="cuda", generator=gen) < 0.2, -1, 1).to(torch.int8)
    return [q, k, v, k2, q2, g2, g, beta], sign


def run(fn, inputs, sign, seed):
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    o = fn(*leaves, sign)
    do = torch.randn(o.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)).to(o.dtype)
    o.backward(do.to(o.dtype))
    return o.float(), [t.grad.float() for t in leaves]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


NAMES = "q k v k2 q2 g2 g beta".split()


@pytest.mark.parametrize("E", [1, 2, 4])
@pytest.mark.parametrize("signed", [False, True])
def test_torch_pipeline_matches_recurrence(E, signed):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (Triton)")
    inputs, sign = make_inputs(1, 128, 2, E, signed, seed=E)
    o_p, g_p = run(lambda *a: chunk_kda_joint(*a[:8], sign=a[8], backend="torch"), inputs, sign, 3)
    o_r, g_r = run(lambda *a: naive_kda_joint(*[t.float() for t in a[:8]], sign=a[8]), inputs, sign, 3)
    assert rel(o_p, o_r) < 1e-2
    for name, a, b in zip(NAMES, g_p, g_r):
        assert rel(a, b) < 1e-2, f"grad {name}: {rel(a, b):.3e}"


CASES = [(2, 256, 4, E, s, gs) for E in (1, 2, 4, 8) for s in (False, True) for gs in (0.5,)] + [
    (1, 512, 2, 4, True, 8.0), (2, 4096, 2, 2, False, 0.05)]


@pytest.mark.parametrize("B,T,H,E,signed,gate_scale", CASES)
def test_cute_against_torch_pipeline(B, T, H, E, signed, gate_scale):
    if not _hopper():
        pytest.skip("the CuTe kernels are built for sm90")
    inputs, sign = make_inputs(B, T, H, E, signed, seed=E * 100 + T, gate_scale=gate_scale)
    o_t, g_t = run(lambda *a: chunk_kda_joint(*a[:8], sign=a[8], backend="torch"), inputs, sign, 7)
    o_c, g_c = run(lambda *a: chunk_kda_joint(*a[:8], sign=a[8], backend="cute"), inputs, sign, 7)
    assert rel(o_c, o_t) < 2e-2, f"forward {rel(o_c, o_t):.3e}"
    for name, a, b in zip(NAMES, g_c, g_t):
        assert rel(a, b) < 5e-2, f"grad {name}: {rel(a, b):.3e}"


@pytest.mark.skipif(not _hopper(), reason="needs sm90")
@pytest.mark.parametrize("gate_scale", [0.5, 4.0])
def test_gram_bwd_fla_matches_triton(gate_scale):
    """flash-linear-attention's intra-chunk backward == the Triton Gram backward."""
    from cute_triadic_gdn.ops import kda_triton as kt

    if kt._fla_intra_bwd() is None:
        pytest.skip("flash-linear-attention not importable")
    torch.manual_seed(0)
    B, T, H, D = 2, 256, 3, 128
    qs = torch.randn(B, T, H, D, device="cuda").div(D ** 0.5).bfloat16()
    ks = torch.randn(B, T, H, D, device="cuda").div(D ** 0.5).bfloat16()
    g = -gate_scale * torch.rand(B, T, H, D, device="cuda") / 16
    gd = g.view(B, T // 64, 64, H, D).cumsum(2).view(B, T, H, D)
    dpkk = torch.randn(B, T, H, 64, device="cuda")
    dpq = torch.randn(B, T, H, 64, device="cuda")
    ref = kt.gram_bwd_triton(qs, ks, gd, dpkk, dpq)
    got = kt.gram_bwd(qs, ks, gd, dpkk, dpq)
    for name, a, b in zip(("dq", "dk", "dg"), got, ref):
        err = (a - b).norm() / b.norm()
        assert err < 2e-2, (name, float(err))
