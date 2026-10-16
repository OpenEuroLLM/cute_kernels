"""The CuTe stages of Triadic KDA on Hopper (see kda_joint.py for the contract).

forward:  slice cumsum (GDN scan) -> kda_masks (A, R', R, Pm) -> kda_fwd (o and the saved states, W, Vd)
backward: kda_bwd_recurrent (state gradients; dqd, dkd, dku and the slice partials) -> kda_bwd_parallel (dPq,
          dPkk, dbeta, the R / R' gradients reduced into dk2, dq2, dg2) -> per-channel decay trace (torch).
"""
import torch

from .gdn_joint import _scan_cute, _value_cols

C = 64


def cute_forward(qd, kd, ku, pkk, pq, lam, v, k2, q2, g2, beta, scale):
    from ..kernels.sm90_kda_masks import kda_masks
    E = k2.shape[-1]
    gc = _scan_cute(g2)
    A, Mp, M, Pm = kda_masks(pkk, pq, k2, q2, gc, beta, scale)
    if E in (1, 2, 4):
        from ..kernels.sm90_kda_fwd import kda_fwd
    else:
        from ..kernels.sm90_kda_fwd_split import kda_fwd_split as kda_fwd
    o, h, W, Vd = kda_fwd(qd, kd, ku, v, k2, q2, gc, beta, A, Pm, lam, scale=scale,
                          value_cols=_value_cols(qd, E, None), save=True)
    return o, (gc, A, Mp, M, Pm, h, W, Vd)


def cute_backward(qd, kd, ku, lam, v, k2, q2, g2, beta, scale, do, saved, pkk, pq):
    from ..kernels.sm90_kda_bwd_recurrent import kda_bwd_recurrent
    from ..kernels.sm90_kda_bwd_parallel import kda_bwd_parallel
    gc, A, Mp, M, Pm, h, W, Vd = saved
    B, T, H, D = qd.shape
    E, NC = k2.shape[-1], T // C
    dqd, dkd, dku, dv, dvd, dgv, dk2v, dgo, dq2o, ds = kda_bwd_recurrent(
        qd, kd, ku, do, A, Pm, gc, k2, q2, beta, lam, h, Vd, scale=scale,
        groups=1 if E == 1 else 2, value_cols=_value_cols(qd, E, None))
    dpq, dpkk, dgc, dk2, dq2, db = kda_bwd_parallel(do, A, Mp, M, pq, pkk, gc, k2, q2, beta, W, Vd, dvd,
                                                     dgv, dk2v, dgo, dq2o, scale=scale)
    # decay over a whole chunk: state rows (e, d) scale by exp(g2_L,e) lam_d.  Its gradient pairs the state at the
    # chunk start (h) with the gradient of the state at its end (ds), row by row.
    trace = (h.float() * ds.float()).sum(-1)                                        # (B, NC, H, E, D)
    egl = gc.view(B, NC, C, H, E)[:, :, -1].exp()                                   # (B, NC, H, E)
    dlam = (egl[..., None] * trace).sum(3)                                          # (B, NC, H, D)
    dgc = dgc.view(B, NC, C, H, E).clone()
    dgc[:, :, -1] += egl * (lam[:, :, :, None, :] * trace).sum(-1)
    dg2 = dgc.flip(2).cumsum(2).flip(2).reshape(B, T, H, E)
    return dqd, dkd, dku, dpkk, dpq, dlam, dv, dk2, dq2, dg2, db
