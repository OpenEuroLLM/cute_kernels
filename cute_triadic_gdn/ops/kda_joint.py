"""Triadic KDA / Complex KDA: the GDN joint pipeline with KDA's per-channel (optionally signed) decay.

The decay of state row (e, d) is sign_{t,d} exp(g2_{t,e} + g_{t,d}) (see kda_reference.py).  Because it is
separable, the chunkwise form factorises: every pair weight is the slice part (the R / R' masks of the GDN
kernels, from k2, q2, g2) times a channel part that only involves q, k and g.  The channel part is formed by
Triton kernels (kda_triton.py) and handed to the CuTe kernels as untied operands:

    qd = qs e^Gd, kd = ks e^Gd     state reads (qs = q sig, ks = k sig; Gd, sig chunk-local cumsum / sign)
    ku = ks sig_L e^(Gd_L - Gd)    state update
    lam = sig_L e^Gd_L             per-channel state decay over a chunk (on top of the slice decay)
    Pkk, Pq                        the channel-decayed key-key / query-key products of the chunk

`untied_forward` is the chunkwise algorithm with those operands as independent inputs, in torch; its autograd
gradients are what the CuTe backward produces, and `_glue_backward` maps them back to q, k, g.
"""
import torch

from . import kda_triton as kt
from .kda_reference import CHUNK as C, _chunks

BACKENDS = ("torch", "cute")


def untied_forward(qd, kd, ku, pkk, pq, lam, v, k2, q2, g2, beta, scale):
    """o (B, T, H, V) fp32 from the untied operands (pkk, pq: (B, T, H, 64); lam: (B, NC, H, D))."""
    B, T, H, D = qd.shape
    E, V, NC = k2.shape[-1], v.shape[-1], T // C
    G2 = _chunks(g2.float()).cumsum(2)
    w, qq, bc, vc = _chunks(k2.float()), _chunks(q2.float()), _chunks(beta.float()), _chunks(v.float())
    dif = (G2[:, :, :, None] - G2[:, :, None, :]).clamp(max=0.0).exp()
    M = torch.einsum("bnthe,bnshe,bntshe->bnhts", w, w, dif)
    Mp = torch.einsum("bnthe,bnshe,bntshe->bnhts", qq, w, dif)
    P = lambda x: _chunks(x.float()).permute(0, 1, 3, 2, 4)                         # (B, NC, H, t, s)
    L = (bc.permute(0, 1, 3, 2)[..., None] * P(pkk) * M).tril(-1)
    A = torch.linalg.inv(torch.eye(C, device=qd.device) + L)
    Pm = scale * (P(pq) * Mp).tril()
    kdc, qdc, kuc = _chunks(kd.float()), _chunks(qd.float()), _chunks(ku.float())
    S = qd.new_zeros(B, H, E, D, V, dtype=torch.float32)
    out = []
    for n in range(NC):
        eg = G2[:, n].exp()
        a, c = w[:, n] * eg, scale * qq[:, n] * eg
        r = w[:, n] * (G2[:, n, -1:] - G2[:, n]).exp()
        U = torch.einsum("bthd,bhedv->bthev", kdc[:, n], S)
        Uq = torch.einsum("bthd,bhedv->bthev", qdc[:, n], S)
        W = vc[:, n] - torch.einsum("bthe,bthev->bthv", a, U)
        Vd = torch.einsum("bhts,bshv->bthv", A[:, n], bc[:, n][..., None] * W)
        out.append(torch.einsum("bthe,bthev->bthv", c, Uq) + torch.einsum("bhts,bshv->bthv", Pm[:, n], Vd))
        decay = G2[:, n, -1].exp()[..., None] * lam[:, n].float()[:, :, None, :]
        S = decay[..., None] * S + torch.einsum("bthe,bthd,bthv->bhedv", r, kuc[:, n], Vd)
    return torch.cat(out, 1)


def _torch_stages(prepped, v, k2, q2, g2, beta, scale, do):
    """The CuTe stages' contract, by autograd of `untied_forward`: o and the gradients of every untied operand
    (dqd, dkd, dku, dpkk, dpq, dlam, dv, dk2, dq2, dg2, dbeta)."""
    qd, kd, ku, pkk, pq, lam = prepped
    leaves = [t.detach().float().requires_grad_() for t in (qd, kd, ku, pkk, pq, lam, v, k2, q2, g2, beta)]
    with torch.enable_grad():
        o = untied_forward(*leaves, scale)
        grads = torch.autograd.grad(o, leaves, do.float()) if do is not None else None
    return o.detach(), grads


def _glue_backward(sig, gd, qs, ks, qd, kd, ku, lam, dqd, dkd, dku, dqs_g, dks_g, dgd_g, dlam):
    """Gradients of q, k and the channel log-decay g from the untied ones (fp32, (B, T, H, D))."""
    B, T, H, D = qs.shape
    eg = gd.exp()
    gl = _chunks(gd)[:, :, -1:]
    sl = _chunks(sig)[:, :, -1:]
    dqs = dqs_g + dqd * eg
    dks = dks_g + dkd * eg + (_chunks(dku) * sl * (gl - _chunks(gd)).exp()).reshape(B, T, H, D)
    kuf = ku.float()
    dgd = dgd_g + dqd * qd.float() + dkd * kd.float() - dku * kuf
    # Gd_L enters ku (all rows of the chunk) and lam
    tail = (_chunks(dku * kuf)).sum(2) + dlam * lam                                  # (B, NC, H, D)
    dgd = _chunks(dgd).clone()
    dgd[:, :, -1] += tail
    dg = dgd.flip(2).cumsum(2).flip(2).reshape(B, T, H, D)                          # through the cumsum
    return dqs * sig, dks * sig, dg


class _KdaJoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k2, q2, g2, g, beta, sign, scale, backend):
        qs, ks, qd, kd, ku, gd, lam = kt.prep(q, k, g, sign)
        pkk, pq = kt.gram_fwd(qs, ks, gd)
        if backend == "torch":
            o, _ = _torch_stages((qd, kd, ku, pkk, pq, lam), v, k2, q2, g2, beta, scale, None)
            saved_fwd = ()
        else:
            from .kda_cute import cute_forward
            o, saved_fwd = cute_forward(qd, kd, ku, pkk, pq, lam, v, k2, q2, g2, beta, scale)
        ctx.save_for_backward(q, k, v, k2, q2, g2, g, beta, sign, *saved_fwd)
        ctx.scale, ctx.backend = scale, backend
        return o.to(v.dtype)

    @staticmethod
    def backward(ctx, do):
        q, k, v, k2, q2, g2, g, beta, sign, *saved_fwd = ctx.saved_tensors
        qs, ks, qd, kd, ku, gd, lam = kt.prep(q, k, g, sign)                     # recomputed: cheap
        pkk, pq = kt.gram_fwd(qs, ks, gd)
        if ctx.backend == "torch":
            _, gr = _torch_stages((qd, kd, ku, pkk, pq, lam), v, k2, q2, g2, beta, ctx.scale, do)
            dqd, dkd, dku, dpkk, dpq, dlam, dv, dk2, dq2, dg2, db = gr
        else:
            from .kda_cute import cute_backward
            dqd, dkd, dku, dpkk, dpq, dlam, dv, dk2, dq2, dg2, db = cute_backward(
                qd, kd, ku, lam, v, k2, q2, g2, beta, ctx.scale, do.contiguous(), saved_fwd, pkk, pq)
        dqs_g, dks_g, dgd_g = kt.gram_bwd(qs, ks, gd, dpkk.float(), dpq.float())
        dq, dk, dg = kt.glue_bwd(sign, qd, kd, ku, gd, lam, dqd, dkd, dku, dlam, dqs_g, dks_g, dgd_g, q.dtype)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dk2.float(), dq2.float(), dg2.float(),
                dg, db.float(), None, None, None)


def chunk_kda_joint(q, k, v, k2, q2, g2, g, beta, sign=None, scale=None, backend=None):
    """Triadic KDA (sign None) / Complex KDA (sign int8 in {-1, +1}) with autograd.

    q, k, v (B, T, H, 128) bf16 (q, k unit norm); k2, q2 (B, T, H, E) f32 (unit norm over E); g2 (B, T, H, E)
    f32 slice log-decay <= 0; g (B, T, H, 128) f32 channel log-decay <= 0; beta (B, T, H) f32 write strength
    (Complex KDA: up to 2); T % 64 == 0.  backend: "cute" (Hopper) or "torch" (the same pipeline with the CuTe
    stages in torch: the reference the kernels are tested against).
    """
    B, T, H, D = q.shape
    assert T % C == 0 and D == 128 and v.shape[-1] == 128
    if backend is None:
        backend = "cute" if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] == 9 else "torch"
    assert backend in BACKENDS
    scale = D ** -0.5 if scale is None else float(scale)
    args = [t.contiguous() for t in (q, k, v, k2, q2, g2, g, beta)]
    return _KdaJoint.apply(*args, sign.contiguous() if sign is not None else None, scale, backend)
