"""Triton kernels for the channel-gate (KDA) parts of Triadic KDA, around the CuTe recurrence.

Per chunk of 64 tokens and head, with Gd the chunk-local cumsum of the channel log-decay g (<= 0) and sig the
chunk-local cumulative sign (Complex KDA; +1 for KDA):

  prep         qs = q sig, ks = k sig (bf16, exact), qd = qs e^Gd and kd = ks e^Gd (state reads),
               ku = ks sig_L e^(Gd_L - Gd) (state update), lam = sig_L e^Gd_L (per-channel chunk decay), Gd.
  gram_fwd     Pkk[t, s] = sum_d ks_td ks_sd e^(Gd_td - Gd_sd)  (s < t)  and  Pq[t, s] (s <= t, qs on the left)
  gram_bwd     the gradients of those products with respect to qs, ks and Gd.
  glue_bwd     dq, dk and dg from the gradients of every untied operand (qd, kd, ku, lam, the products):
               the chain rule through the exponentials, the signs and the chunk-local cumsum.

Every exponential has a non-positive argument: rows are taken in blocks of 16, an off-diagonal block pair
(I > J) factors e^(Gd_t - Gd_s) = e^(Gd_t - Gd_r) e^(Gd_r - Gd_s) at the first row r of block I (both <= 1),
and a diagonal block forms the pairwise exponentials directly. So no gate bound is needed.
"""
import torch
import triton
import triton.language as tl

C = 64
BS = 16


@triton.jit
def _prep_kernel(q, k, g, sign, qs_out, ks_out, qd_out, kd_out, ku_out, gd_out, lam_out,
                 T, H, D: tl.constexpr, C: tl.constexpr, HAS_SIGN: tl.constexpr):
    n, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    rows = tl.arange(0, C)
    cols = tl.arange(0, D)
    off = ((b * T + n * C + rows[:, None]) * H + h) * D + cols[None, :]
    gd = tl.cumsum(tl.load(g + off).to(tl.float32), axis=0)
    qv = tl.load(q + off).to(tl.float32)
    kv = tl.load(k + off).to(tl.float32)
    if HAS_SIGN:
        neg = (tl.load(sign + off) < 0).to(tl.int32)
        sig = 1.0 - 2.0 * (tl.cumsum(neg, axis=0) % 2).to(tl.float32)
        qv = qv * sig
        kv = kv * sig
        sig_l = tl.sum(tl.where(rows[:, None] == C - 1, sig, 0.0), axis=0)
    else:
        sig_l = tl.full((D,), 1.0, tl.float32)
    gd_l = tl.sum(tl.where(rows[:, None] == C - 1, gd, 0.0), axis=0)
    eg = tl.exp(gd)
    tl.store(qs_out + off, qv.to(qs_out.dtype.element_ty))
    tl.store(ks_out + off, kv.to(ks_out.dtype.element_ty))
    tl.store(qd_out + off, (qv * eg).to(qd_out.dtype.element_ty))
    tl.store(kd_out + off, (kv * eg).to(kd_out.dtype.element_ty))
    tl.store(ku_out + off, (kv * sig_l[None, :] * tl.exp(gd_l[None, :] - gd)).to(ku_out.dtype.element_ty))
    tl.store(gd_out + off, gd)
    tl.store(lam_out + ((b * (T // C) + n) * H + h) * D + cols, sig_l * tl.exp(gd_l))


_GRAM_CONFIGS = [
    triton.Config({"BS": bs, "BD": bd}, num_warps=w)
    for bs in (16, 32) for bd in (16, 32, 64) for w in (4, 8)
    if bs * bs * bd <= 32 * 32 * 32  # the diagonal block's [BS, BS, BD] exponentials
]


@triton.autotune(configs=_GRAM_CONFIGS, key=["T", "H", "D"])
@triton.jit
def _gram_fwd_kernel(qs, ks, gd, pkk, pq, T, H, D: tl.constexpr, C: tl.constexpr, BS: tl.constexpr,
                     BD: tl.constexpr):
    """Rows of block X: off-diagonal blocks J < X by the factored exponentials (tf32 dots), the diagonal block by
    pairwise exponentials (shared by both products), zeros above."""
    n, bh, x = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = bh // H, bh % H
    qh, kh, gh = qs + h * D, ks + h * D, gd + h * D
    t = tl.arange(0, BS)
    rowx = b * T + n * C + x * BS
    out_rows = ((rowx + t[:, None]) * H + h) * C
    keep = t[:, None] >= t[None, :]
    acc_q = tl.zeros((BS, BS), tl.float32)
    acc_k = tl.zeros((BS, BS), tl.float32)
    for d0 in tl.static_range(0, D, BD):
        dd = d0 + tl.arange(0, BD)
        offx = (rowx + t[:, None]) * H * D + dd[None, :]
        gx = tl.load(gh + offx)
        qx = tl.load(qh + offx).to(tl.float32)
        kx = tl.load(kh + offx).to(tl.float32)
        ex = tl.exp(tl.minimum(gx[:, None, :] - gx[None, :, :], 0.0))
        acc_q += tl.sum(qx[:, None, :] * kx[None, :, :] * ex, axis=2)
        acc_k += tl.sum(kx[:, None, :] * kx[None, :, :] * ex, axis=2)
    cols = x * BS + t[None, :]
    tl.store(pq + out_rows + cols, tl.where(keep, acc_q, 0.0))
    tl.store(pkk + out_rows + cols, tl.where(t[:, None] > t[None, :], acc_k, 0.0))
    dd = tl.arange(0, D)
    offx = (rowx + t[:, None]) * H * D + dd[None, :]
    gr = tl.load(gh + rowx * H * D + dd)              # Gd at the reference row (first of X)
    ei = tl.exp(tl.load(gh + offx) - gr[None, :])
    lq = tl.load(qh + offx).to(tl.float32) * ei
    lk = tl.load(kh + offx).to(tl.float32) * ei
    for j in tl.range(0, x):
        rowj = b * T + n * C + j * BS
        offj = (rowj + t[:, None]) * H * D + dd[None, :]
        rj = tl.load(kh + offj).to(tl.float32) * tl.exp(gr[None, :] - tl.load(gh + offj))
        cols = j * BS + t[None, :]
        tl.store(pq + out_rows + cols, tl.dot(lq, tl.trans(rj)))
        tl.store(pkk + out_rows + cols, tl.dot(lk, tl.trans(rj)))
    for j in tl.range(x + 1, C // BS):
        cols = j * BS + t[None, :]
        zero = tl.zeros((BS, BS), tl.float32)
        tl.store(pq + out_rows + cols, zero)
        tl.store(pkk + out_rows + cols, zero)


@triton.autotune(
    configs=[triton.Config({"BS": bs}, num_warps=w, num_stages=st)
             for bs in (16, 32) for w in (2, 4, 8) for st in (1, 2)],
    key=["T", "H", "D"],
)
@triton.jit
def _gram_fwd_rows_kernel(qs, ks, gd, pkk, pq, T, H, D: tl.constexpr, C: tl.constexpr, BS: tl.constexpr):
    """As _gram_fwd_kernel, with the diagonal block one row at a time (a [BS, D] exponential tile per row
    instead of a [BS, BS, BD] one) over the whole head dim."""
    n, bh, x = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = bh // H, bh % H
    qh, kh, gh = qs + h * D, ks + h * D, gd + h * D
    t = tl.arange(0, BS)
    dd = tl.arange(0, D)
    rowx = b * T + n * C + x * BS
    out_rows = ((rowx + t[:, None]) * H + h) * C
    offx = (rowx + t[:, None]) * H * D + dd[None, :]
    gx = tl.load(gh + offx)
    kx = tl.load(kh + offx).to(tl.float32)
    # diagonal block: row r against the block's rows s
    for r in tl.static_range(BS):
        offr = (rowx + r) * H * D + dd
        gr = tl.load(gh + offr)
        ke = kx * tl.exp(tl.minimum(gr[None, :] - gx, 0.0))
        prow = ((rowx + r) * H + h) * C + x * BS + t
        tl.store(pq + prow, tl.where(t <= r, tl.sum(tl.load(qh + offr).to(tl.float32)[None, :] * ke, axis=1), 0.0))
        tl.store(pkk + prow, tl.where(t < r, tl.sum(tl.load(kh + offr).to(tl.float32)[None, :] * ke, axis=1), 0.0))
    # off-diagonal blocks J < X: factored at the first row of X (both factors <= 1), tensor-core dots
    gr = tl.load(gh + rowx * H * D + dd)
    ei = tl.exp(gx - gr[None, :])
    lq = tl.load(qh + offx).to(tl.float32) * ei
    lk = kx * ei
    for j in tl.range(0, x):
        rowj = b * T + n * C + j * BS
        offj = (rowj + t[:, None]) * H * D + dd[None, :]
        rj = tl.load(kh + offj).to(tl.float32) * tl.exp(gr[None, :] - tl.load(gh + offj))
        cols = j * BS + t[None, :]
        tl.store(pq + out_rows + cols, tl.dot(lq, tl.trans(rj)))
        tl.store(pkk + out_rows + cols, tl.dot(lk, tl.trans(rj)))
    for j in tl.range(x + 1, C // BS):
        cols = j * BS + t[None, :]
        zero = tl.zeros((BS, BS), tl.float32)
        tl.store(pq + out_rows + cols, zero)
        tl.store(pkk + out_rows + cols, zero)


@triton.autotune(configs=_GRAM_CONFIGS, key=["T", "H", "D"])
@triton.jit
def _gram_bwd_kernel(qs, ks, gd, dpkk, dpq, dqs, dks, dgd, T, H, D: tl.constexpr, C: tl.constexpr,
                     BS: tl.constexpr, BD: tl.constexpr):
    """Block X of rows, one BD-wide channel slice at a time: its row role (t in X; dqs, d(ks as the left key)) and
    column role (s in X; d(ks as the right key)), then dGd = qs*dqs + ks*(dk_left - dk_right) by the shift identity
    of e^(Gd_t - Gd_s)."""
    n, bh, x = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = bh // H, bh % H
    qh, kh, gh = qs + h * D, ks + h * D, gd + h * D
    t = tl.arange(0, BS)
    rowx = b * T + n * C + x * BS
    pdiag = ((rowx + t[:, None]) * H + h) * C + x * BS + t[None, :]
    pq_d = tl.where(t[:, None] >= t[None, :], tl.load(dpq + pdiag), 0.0)
    pk_d = tl.where(t[:, None] > t[None, :], tl.load(dpkk + pdiag), 0.0)
    for d0 in tl.static_range(0, D, BD):
        dd = d0 + tl.arange(0, BD)
        offx = (rowx + t[:, None]) * H * D + dd[None, :]
        gx = tl.load(gh + offx)
        grx = tl.load(gh + rowx * H * D + dd)
        qx = tl.load(qh + offx).to(tl.float32)
        kx = tl.load(kh + offx).to(tl.float32)
        dq = tl.zeros((BS, BD), tl.float32)
        dkl = tl.zeros((BS, BD), tl.float32)
        dkr = tl.zeros((BS, BD), tl.float32)
        # row role, off-diagonal: s in J < X, reference row = first row of X
        for j in tl.range(0, x):
            rowj = b * T + n * C + j * BS
            offj = (rowj + t[:, None]) * H * D + dd[None, :]
            rj = tl.load(kh + offj).to(tl.float32) * tl.exp(grx[None, :] - tl.load(gh + offj))
            pidx = ((rowx + t[:, None]) * H + h) * C + j * BS + t[None, :]
            dq += tl.dot(tl.load(dpq + pidx), rj)
            dkl += tl.dot(tl.load(dpkk + pidx), rj)
        ex = tl.exp(gx - grx[None, :])
        dq = dq * ex
        dkl = dkl * ex
        # column role, off-diagonal: t in I > X, reference row = first row of I
        for i in tl.range(x + 1, C // BS):
            rowi = b * T + n * C + i * BS
            offi = (rowi + t[:, None]) * H * D + dd[None, :]
            gri = tl.load(gh + rowi * H * D + dd)
            ei = tl.exp(tl.load(gh + offi) - gri[None, :])
            pidx = ((rowi + t[:, None]) * H + h) * C + x * BS + t[None, :]          # [t in I, s in X]
            part = tl.dot(tl.trans(tl.load(dpq + pidx)), tl.load(qh + offi).to(tl.float32) * ei)
            part += tl.dot(tl.trans(tl.load(dpkk + pidx)), tl.load(kh + offi).to(tl.float32) * ei)
            dkr += part * tl.exp(gri[None, :] - gx)
        # diagonal block: pairwise exponentials
        e3 = tl.exp(tl.minimum(gx[:, None, :] - gx[None, :, :], 0.0))              # [t, s, d]
        dq += tl.sum(pq_d[:, :, None] * kx[None, :, :] * e3, axis=1)
        dkl += tl.sum(pk_d[:, :, None] * kx[None, :, :] * e3, axis=1)
        dkr += tl.sum((pq_d[:, :, None] * qx[:, None, :] + pk_d[:, :, None] * kx[:, None, :]) * e3, axis=0)
        tl.store(dqs + h * D + offx, dq)
        tl.store(dks + h * D + offx, dkl + dkr)
        tl.store(dgd + h * D + offx, qx * dq + kx * (dkl - dkr))


@triton.jit
def _glue_bwd_kernel(sign, qd, kd, ku, gd, lam, dqd, dkd, dku, dlam, dqs_g, dks_g, dgd_g, dq, dk, dg,
                     T, H, D: tl.constexpr, C: tl.constexpr, HAS_SIGN: tl.constexpr):
    n, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    rows = tl.arange(0, C)
    cols = tl.arange(0, D)
    off = ((b * T + n * C + rows[:, None]) * H + h) * D + cols[None, :]
    loff = ((b * (T // C) + n) * H + h) * D + cols
    g = tl.load(gd + off)
    last = rows[:, None] == C - 1
    gl = tl.sum(tl.where(last, g, 0.0), axis=0)
    eg = tl.exp(g)
    if HAS_SIGN:
        neg = (tl.load(sign + off) < 0).to(tl.int32)
        sig = 1.0 - 2.0 * (tl.cumsum(neg, axis=0) % 2).to(tl.float32)
        sl = tl.sum(tl.where(last, sig, 0.0), axis=0)
    else:
        sig = tl.full((C, D), 1.0, tl.float32)
        sl = tl.full((D,), 1.0, tl.float32)
    a_qd = tl.load(dqd + off)
    a_kd = tl.load(dkd + off)
    a_ku = tl.load(dku + off)
    v_ku = tl.load(ku + off).to(tl.float32)
    dqs = tl.load(dqs_g + off) + a_qd * eg
    dks = tl.load(dks_g + off) + a_kd * eg + a_ku * sl[None, :] * tl.exp(gl[None, :] - g)
    ddg = (tl.load(dgd_g + off) + a_qd * tl.load(qd + off).to(tl.float32) + a_kd * tl.load(kd + off).to(tl.float32)
           - a_ku * v_ku)
    # Gd_L enters ku (every row) and lam
    tail = tl.sum(a_ku * v_ku, axis=0) + tl.load(dlam + loff) * tl.load(lam + loff)
    ddg = ddg + tl.where(last, tail[None, :], 0.0)
    tl.store(dg + off, tl.cumsum(ddg, axis=0, reverse=True))
    tl.store(dq + off, (dqs * sig).to(dq.dtype.element_ty))
    tl.store(dk + off, (dks * sig).to(dk.dtype.element_ty))


def glue_bwd(sign, qd, kd, ku, gd, lam, dqd, dkd, dku, dlam, dqs_g, dks_g, dgd_g, out_dtype):
    """(dq, dk) in out_dtype and dg fp32, (B, T, H, D)."""
    B, T, H, D = qd.shape
    dq = torch.empty(B, T, H, D, device=qd.device, dtype=out_dtype)
    dk = torch.empty_like(dq)
    dg = torch.empty(B, T, H, D, device=qd.device, dtype=torch.float32)
    c = lambda t: t.float().contiguous()
    _glue_bwd_kernel[(T // C, B * H)](sign if sign is not None else gd, qd, kd, ku, gd, c(lam), c(dqd), c(dkd), c(dku),
                                      c(dlam), dqs_g, dks_g, dgd_g, dq, dk, dg, T, H, D, C, sign is not None,
                                      num_warps=8)
    return dq, dk, dg


def prep(q, k, g, sign=None):
    """(qs, ks, qd, kd, ku) bf16 (B, T, H, D), Gd fp32 (B, T, H, D), lam fp32 (B, NC, H, D)."""
    B, T, H, D = q.shape
    assert T % C == 0 and D == 128
    outs = [torch.empty_like(q) for _ in range(5)]
    gd = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
    lam = torch.empty(B, T // C, H, D, device=q.device, dtype=torch.float32)
    _prep_kernel[(T // C, B * H)](q, k, g, sign if sign is not None else g, *outs, gd, lam, T, H, D, C,
                                  sign is not None, num_warps=8)
    return (*outs, gd, lam)


def gram_fwd(qs, ks, gd):
    """Pkk (strictly lower) and Pq (lower) as (B, T, H, 64) fp32: row t of each chunk, column s."""
    B, T, H, D = qs.shape
    pkk = torch.empty(B, T, H, C, device=qs.device, dtype=torch.float32)
    pq = torch.empty_like(pkk)
    grid = lambda meta: (T // C, B * H, C // meta["BS"])  # noqa: E731
    _gram_fwd_rows_kernel[grid](qs, ks, gd, pkk, pq, T, H, D, C)
    return pkk, pq


def gram_fwd_triton(qs, ks, gd):
    """The previous Gram forward (diagonal block by [BS, BS, BD] pairwise exponentials)."""
    B, T, H, D = qs.shape
    pkk = torch.empty(B, T, H, C, device=qs.device, dtype=torch.float32)
    pq = torch.empty_like(pkk)
    grid = lambda meta: (T // C, B * H, C // meta["BS"])  # noqa: E731
    _gram_fwd_kernel[grid](qs, ks, gd, pkk, pq, T, H, D, C)
    return pkk, pq


_LOG2E = 1.4426950408889634


def _fla_intra_bwd():
    """flash-linear-attention's KDA intra-chunk backward with CUTE_KDA_GRAM_BWD=fla, else None
    (the default: the Triton kernel below is 1.6x faster at the speed suite's shape on H100,
    1.41 vs 2.20 ms, bench_gram.py)."""
    import os

    if os.environ.get("CUTE_KDA_GRAM_BWD", "triton") != "fla":
        return None
    try:
        from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    except ImportError:
        return None
    return chunk_kda_bwd_intra


def gram_bwd(qs, ks, gd, dpkk, dpq):
    """(dqs, dks, dGd) fp32 (B, T, H, D) from the gradients of Pkk and Pq (B, T, H, 64) fp32.

    The products are those of KDA's intra-chunk attention (beta = 1, scale = 1), so
    flash-linear-attention's intra-chunk backward can compute them (CUTE_KDA_GRAM_BWD=fla). It takes the gate in base 2
    (times log2(e)) but returns the gate gradient for the natural-log gate, and it reads the whole
    row of dAqk / dAkk, so the entries outside the products' triangles are zeroed first. The Triton kernel below
    is the default (CUTE_KDA_GRAM_BWD=fla selects FLA's)."""
    fla_bwd = _fla_intra_bwd()
    if fla_bwd is not None:
        B, T, H, D = qs.shape
        t = torch.arange(C, device=qs.device)
        lower = (t[:, None] >= t[None, :]).repeat(T // C, 1)[None, :, None, :]
        strict = (t[:, None] > t[None, :]).repeat(T // C, 1)[None, :, None, :]
        zeros = torch.zeros(B, T, H, D, device=qs.device, dtype=torch.float32)
        beta = torch.ones(B, T, H, device=qs.device, dtype=torch.float32)
        dq, dk, _, dg = fla_bwd(
            q=qs, k=ks, g=gd * _LOG2E, beta=beta,
            dAqk=dpq.masked_fill(~lower, 0.0), dAkk=dpkk.masked_fill(~strict, 0.0),
            dq=zeros, dk=zeros.clone(), db=torch.zeros_like(beta), dg=zeros.clone(), chunk_size=C,
        )
        return dq.float(), dk.float(), dg.float()
    return gram_bwd_triton(qs, ks, gd, dpkk, dpq)


def gram_bwd_triton(qs, ks, gd, dpkk, dpq):
    """The Triton Gram backward (see gram_bwd)."""
    B, T, H, D = qs.shape
    dqs = torch.empty(B, T, H, D, device=qs.device, dtype=torch.float32)
    dks, dgd = torch.empty_like(dqs), torch.empty_like(dqs)
    grid = lambda meta: (T // C, B * H, C // meta["BS"])  # noqa: E731
    _gram_bwd_kernel[grid](qs, ks, gd, dpkk.contiguous(), dpq.contiguous(), dqs, dks, dgd, T, H, D, C)
    return dqs, dks, dgd
