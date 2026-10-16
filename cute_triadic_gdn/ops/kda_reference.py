"""Triadic KDA / Complex KDA: torch references (the test oracles).

Per head, a third-order state of E slices S_e (D x V) with a separable, optionally signed decay

    S_{e,d,:} <- sign_{t,d} exp(g2_{t,e} + g_{t,d}) S_{e,d,:}      (KDA: no sign; Complex KDA: sign in {+-1})
    S         <- S + beta_t kappa_t (v_t - S^T kappa_t)^T,           kappa_t = k2_t (x) k_t
    o_t        = S^T (q2_t (x) q_t) * scale

`naive_kda_joint` is the token recurrence; `chunk_kda_joint_torch` the chunkwise form the kernels implement
(see kda_joint.py for the algebra), differentiable through autograd.
"""
import torch

CHUNK = 64


def naive_kda_joint(q, k, v, k2, q2, g2, g, beta, sign=None, scale=None):
    """Token-by-token recurrence over S (B, H, E, D, V) in fp32; differentiable."""
    B, T, H, D = q.shape
    E, V = k2.shape[-1], v.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    S = q.new_zeros(B, H, E, D, V, dtype=torch.float32)
    out = []
    for t in range(T):
        log_a = g2[:, t].float().unsqueeze(-1) + g[:, t].float().unsqueeze(-2)        # (B, H, E, D)
        alpha = log_a.exp()
        if sign is not None:
            alpha = alpha * sign[:, t].float().unsqueeze(-2)
        S = S * alpha.unsqueeze(-1)
        kappa = k2[:, t].float().unsqueeze(-1) * k[:, t].float().unsqueeze(-2)
        pred = torch.einsum("bhed,bhedv->bhv", kappa, S)
        upd = beta[:, t].float()[..., None] * (v[:, t].float() - pred)
        S = S + kappa.unsqueeze(-1) * upd[:, :, None, None, :]
        qq = q2[:, t].float().unsqueeze(-1) * q[:, t].float().unsqueeze(-2)
        out.append(torch.einsum("bhed,bhedv->bhv", qq, S) * scale)
    return torch.stack(out, 1)


def _chunks(t, C=CHUNK):
    B, T = t.shape[:2]
    return t.reshape(B, T // C, C, *t.shape[2:])


def prep_torch(q, k, g, sign=None, C=CHUNK):
    """The KDA-specific per-chunk quantities, fp32, (B, NC, C, H, X):
    Gd chunk-local cumsum of g, sig the chunk-local cumulative sign, qs/ks = q/k * sig,
    qd = qs e^Gd, kd = ks e^Gd (state reads), ku = ks sig_L e^(Gd_L - Gd) (state update),
    lam = sig_L e^Gd_L (B, NC, H, D) (per-channel state decay over the chunk),
    Pkk / Pq (B, NC, H, C, C): the decayed key-key / query-key products (lower triangle)."""
    Gd = _chunks(g.float(), C).cumsum(2)
    if sign is not None:
        neg = _chunks((sign < 0).to(torch.int32), C).cumsum(2) % 2
        sig = 1.0 - 2.0 * neg.float()
    else:
        sig = torch.ones_like(Gd)
    qs, ks = _chunks(q.float(), C) * sig, _chunks(k.float(), C) * sig
    eG = Gd.exp()
    GL, sL = Gd[:, :, -1:], sig[:, :, -1:]
    qd, kd = qs * eG, ks * eG
    ku = ks * sL * (GL - Gd).exp()
    lam = (sL * GL.exp())[:, :, 0]                                                  # (B, NC, H, D)
    diff = (Gd[:, :, :, None] - Gd[:, :, None, :]).clamp(max=0.0).exp()             # (B, NC, t, s, H, D)
    Pkk = torch.einsum("bntshd,bnthd,bnshd->bnhts", diff, ks, ks)
    Pq = torch.einsum("bntshd,bnthd,bnshd->bnhts", diff, qs, ks)
    tri = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    Pkk = Pkk.masked_fill(~tri.tril(-1), 0.0)
    Pq = Pq.masked_fill(~tri, 0.0)
    return dict(Gd=Gd, sig=sig, qs=qs, ks=ks, qd=qd, kd=kd, ku=ku, lam=lam, Pkk=Pkk, Pq=Pq)


def chunk_kda_joint_torch(q, k, v, k2, q2, g2, g, beta, sign=None, scale=None, C=CHUNK):
    """The chunkwise algorithm of the kernels in fp32 torch (differentiable). T % C == 0."""
    B, T, H, D = q.shape
    E, V = k2.shape[-1], v.shape[-1]
    NC = T // C
    scale = D ** -0.5 if scale is None else scale
    p = prep_torch(q, k, g, sign, C)
    G2 = _chunks(g2.float(), C).cumsum(2)                                            # (B, NC, C, H, E)
    w, qq = _chunks(k2.float(), C), _chunks(q2.float(), C)
    bc = _chunks(beta.float(), C)                                                    # (B, NC, C, H)
    vc = _chunks(v.float(), C)
    dif = (G2[:, :, :, None] - G2[:, :, None, :]).clamp(max=0.0).exp()              # (B, NC, t, s, H, E)
    M = torch.einsum("bnthe,bnshe,bntshe->bnhts", w, w, dif)
    Mp = torch.einsum("bnthe,bnshe,bntshe->bnhts", qq, w, dif)
    I = torch.eye(C, device=q.device)
    L = (bc.permute(0, 1, 3, 2)[..., None] * p["Pkk"] * M).tril(-1)
    A = torch.linalg.inv(I + L)                                                      # (B, NC, H, C, C)
    Pm = scale * (p["Pq"] * Mp).tril()
    S = q.new_zeros(B, H, E, D, V, dtype=torch.float32)
    out = []
    for n in range(NC):
        a = w[:, n] * G2[:, n].exp()                                                 # (B, C, H, E)
        c = scale * qq[:, n] * G2[:, n].exp()
        r = w[:, n] * (G2[:, n, -1:] - G2[:, n]).exp()
        U = torch.einsum("bthd,bhedv->bthev", p["kd"][:, n], S)
        Uq = torch.einsum("bthd,bhedv->bthev", p["qd"][:, n], S)
        W = vc[:, n] - torch.einsum("bthe,bthev->bthv", a, U)
        Vd = torch.einsum("bhts,bshv->bthv", A[:, n], bc[:, n][..., None] * W)
        out.append(torch.einsum("bthe,bthev->bthv", c, Uq) + torch.einsum("bhts,bshv->bthv", Pm[:, n], Vd))
        decay = G2[:, n, -1].exp()[..., None] * p["lam"][:, n][:, :, None, :]         # (B, H, E, D)
        S = decay[..., None] * S + torch.einsum("bthe,bthd,bthv->bhedv", r, p["ku"][:, n], Vd)
    return torch.cat(out, 1)
