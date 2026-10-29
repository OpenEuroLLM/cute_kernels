"""cute_ldm against an FP64 ground truth at the production shape (G 16, K 8, D 128, 97 edges,
window 4, 2 edges per recompute group) in every regime of cute_ldm.numerics; the analysis and the
measured numbers are in cute_ldm/NUMERICS.md. `pytest tests/test_ldm_numerics.py`.

The bounds are relative to the BF16 floor (the FP64 result rounded to BF16, ~1.7e-3 normwise) and
sit ~1.5-2x above the measured worst case (GH200, CuTe DSL 4.7.1). The query gradient and the
recomputed reads get separate, wider bounds: they come from the BF16 M the backward unwinds by
subtraction, whose absolute error follows the largest state of the chain (NUMERICS.md)."""
import pytest
import torch

from cute_ldm import numerics, reference


def _hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (9, 10)


# -- the ground truth itself (CPU) ---------------------------------------------------------------
def test_fp64_reference_is_the_weighted_ridge_solution():
    """After L writes, M and P of the FP64 chain are the exact weighted ridge solution and the
    inverse of its regularized Gram matrix (the paper's Eq. 6)."""
    L, n, g, k, d, alpha = 33, 16, 4, 8, 16, 0.7
    steps, _ = numerics.make_steps("aligned_forget", L, n, g, k, d, device="cpu")
    steps = [tuple(t.double() for t in s) for s in steps]
    _, _, kept = reference.chain(steps, alpha, torch.float64, states=[L - 1])
    a = torch.eye(k, dtype=torch.float64).expand(n, g, k, k) / alpha
    b = torch.zeros(n, g, k, d, dtype=torch.float64)
    for khat, _, value, scale, lam in steps:
        kbar, vbar, lam = khat * scale[..., None], value * scale[..., None], lam[:, None, None]
        a = lam * a + kbar[..., :, None] * kbar[..., None, :]
        b = lam * b + kbar[..., :, None] * vbar[..., None, :]
    m, p = kept[L - 1]
    torch.testing.assert_close(m, torch.linalg.solve(a, b), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(p, torch.linalg.inv(a), rtol=1e-10, atol=1e-12)


def test_fp64_reference_gradients_match_finite_differences():
    """The FP64 chain's autograd against central differences of a scalar loss (one entry of
    every input of every edge)."""
    L, n, g, k, d = 6, 2, 2, 4, 3
    steps, cot = numerics.make_steps("nominal", L, n, g, k, d, device="cpu")
    steps = [[t.double() for t in s] for s in steps]
    cot = [c.double() for c in cot]

    def loss():
        reads, _, _ = reference.chain(steps, 1.0, torch.float64)
        return sum((r * c).sum() for r, c in zip(reads, cot))

    _, grads, _ = reference.chain(steps, 1.0, torch.float64, cot)
    for e in range(L):
        for j in range(5):
            t = steps[e][j]
            idx = (0,) * t.dim()
            old = t[idx].item()
            t[idx] = old + 1e-6
            up = loss()
            t[idx] = old - 1e-6
            down = loss()
            t[idx] = old
            fd = (up - down) / 2e-6
            assert abs(float(grads[e][j][idx]) - float(fd)) < 1e-6 * max(1.0, abs(float(fd))), (e, j)


def test_fp32_needs_a_symmetric_covariance_under_strong_forgetting():
    """The RLS instability: without symmetrization the rounding's asymmetric part of P grows by
    1 / lambda per edge (Megatron's torch paths update P this way); with it FP32 stays at ~1e-7.
    The kernels keep P as a packed triangle and are not affected."""
    L, n, g, k, d = 97, 32, 4, 8, 16
    steps, _ = numerics.make_steps("forget_0.85", L, n, g, k, d, device="cpu")
    ref, _, _ = reference.chain(steps, 1.0, torch.float64)
    plain, _, _ = reference.chain(steps, 1.0, torch.float32)
    sym, _, _ = reference.chain(steps, 1.0, torch.float32, symmetric=True)
    assert numerics.rel(sym[-1], ref[-1]) < 1e-6
    assert numerics.rel(plain[-1], ref[-1]) > 1e-3


# -- the kernels (Hopper / Blackwell) ------------------------------------------------------------
# (max over the edges of the normwise relative error) <= bound x BF16 floor
BOUNDS = dict(read=5, khat=12, value=8, scale=8, forgetting=8)
# the quantities that follow the unwound BF16 M: absolute bounds, per regime
QUERY = dict(nominal=0.04, aligned_forget=0.2, value_growth=0.15)       # others 0.08
RECOMPUTED = dict(nominal=0.06, aligned_forget=0.3, value_growth=0.3)   # others 0.12


@pytest.fixture(scope="module", params=list(numerics.REGIMES))
def analysis(request):
    if not _hopper():
        pytest.skip("cute_ldm needs sm90 / sm100")
    return numerics.analyse(request.param)


def test_reads_and_gradients(analysis):
    floor = analysis["impl"]["bf16_floor"]
    regime = analysis["regime"]
    for impl in ("cute", "cute_recompute"):
        res = analysis["impl"][impl]
        for name, bound in BOUNDS.items():
            worst, ref = max(res[name]), max(floor[name])
            assert worst < bound * ref, (regime, impl, name, worst, ref)
        assert max(res["query"]) < QUERY.get(regime, 0.08), (regime, impl, max(res["query"]))


def test_recomputed_reads(analysis):
    res = analysis["impl"]["cute_recompute"]
    regime = analysis["regime"]
    # the forward reads are the kernels' without recompute, bit for bit
    assert res["read"] == analysis["impl"]["cute"]["read"]
    worst = max(res["recomputed_read"].values())
    assert worst < RECOMPUTED.get(regime, 0.12), (regime, worst)


def test_unwound_state(analysis):
    """The backward's state at every window start: P (FP32, by addition) to ~1e-6; M (BF16, by
    subtraction) within the query gradient's bound."""
    res = analysis["impl"]["cute"]
    regime = analysis["regime"]
    assert max(res["unwound_p"].values()) < 1e-5, regime
    assert max(res["unwound_m"].values()) < QUERY.get(regime, 0.08), regime
    # the forward's committed M stays at the BF16 floor's order at every depth
    assert max(res["fwd_m"].values()) < 4 * max(analysis["impl"]["bf16_floor"]["read"]), regime
