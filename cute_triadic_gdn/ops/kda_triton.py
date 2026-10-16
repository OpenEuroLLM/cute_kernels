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


@triton.jit
def _diag_block(left, right, gd, base, H, D: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr,
                STRICT: tl.constexpr):
    """sum_d left_td right_sd e^(Gd_td - Gd_sd) for t, s in one 16-row block (s <= t, or s < t)."""
    t = tl.arange(0, BS)
    acc = tl.zeros((BS, BS), tl.float32)
    for d0 in tl.static_range(0, D, BD):
        dd = d0 + tl.arange(0, BD)
        off = (base + t[:, None]) * H * D + dd[None, :]
        lt = tl.load(left + off).to(tl.float32)
        rt = tl.load(right + off).to(tl.float32)
        gt = tl.load(gd + off)
        ex = tl.exp(tl.minimum(gt[:, None, :] - gt[None, :, :], 0.0))
        acc += tl.sum(lt[:, None, :] * rt[None, :, :] * ex, axis=2)
    if STRICT:
        keep = t[:, None] > t[None, :]
    else:
        keep = t[:, None] >= t[None, :]
    return tl.where(keep, acc, 0.0)


@triton.jit
def _gram_fwd_kernel(qs, ks, gd, pkk, pq, T, H, D: tl.constexpr, C: tl.constexpr, BS: tl.constexpr,
                     BD: tl.constexpr):
    n, bh, blk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = bh // H, bh % H
    head = qs + h * D
    keyh = ks + h * D
    gdh = gd + h * D
    row0 = b * T + n * C + blk * BS                    # token index of the block's first row
    t = tl.arange(0, BS)
    dd = tl.arange(0, D)
    offi = (row0 + t[:, None]) * H * D + dd[None, :]
    gr = tl.load(gdh + row0 * H * D + dd)              # Gd at the reference row
    gi = tl.load(gdh + offi)
    ei = tl.exp(gi - gr[None, :])
    lq = (tl.load(head + offi).to(tl.float32) * ei)
    lk = (tl.load(keyh + offi).to(tl.float32) * ei)
    out_rows = ((row0 + t[:, None]) * H + h) * C
    for j in tl.range(0, blk):
        rowj = b * T + n * C + j * BS
        offj = (rowj + t[:, None]) * H * D + dd[None, :]
        rj = tl.load(keyh + offj).to(tl.float32) * tl.exp(gr[None, :] - tl.load(gdh + offj))
        cols = j * BS + t[None, :]
        tl.store(pq + out_rows + cols, tl.dot(lq, tl.trans(rj), input_precision="ieee"))
        tl.store(pkk + out_rows + cols, tl.dot(lk, tl.trans(rj), input_precision="ieee"))
    cols = blk * BS + t[None, :]
    tl.store(pq + out_rows + cols, _diag_block(head, keyh, gdh, row0, H, D, BS, 32, False))
    tl.store(pkk + out_rows + cols, _diag_block(keyh, keyh, gdh, row0, H, D, BS, 32, True))
    for j in tl.range(blk + 1, C // BS):
        cols = j * BS + t[None, :]
        zero = tl.zeros((BS, BS), tl.float32)
        tl.store(pq + out_rows + cols, zero)
        tl.store(pkk + out_rows + cols, zero)


@triton.jit
def _gram_bwd_kernel(qs, ks, gd, dpkk, dpq, dqs, dks, dgd, T, H, D: tl.constexpr, C: tl.constexpr,
                     BS: tl.constexpr, BD: tl.constexpr):
    """Block X of rows: its row role (t in X; dqs, d(ks as the left key)) and column role (s in X; d(ks as the
    right key)), then dGd = qs*dqs_q + ks*(dk_left - dk_right) by the shift identity of e^(Gd_t - Gd_s)."""
    n, bh, x = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = bh // H, bh % H
    qh, kh, gh = qs + h * D, ks + h * D, gd + h * D
    t = tl.arange(0, BS)
    dd = tl.arange(0, D)
    rowx = b * T + n * C + x * BS
    offx = (rowx + t[:, None]) * H * D + dd[None, :]
    gx = tl.load(gh + offx)
    grx = tl.load(gh + rowx * H * D + dd)
    dq = tl.zeros((BS, D), tl.float32)
    dkl = tl.zeros((BS, D), tl.float32)
    dkr = tl.zeros((BS, D), tl.float32)
    # row role, off-diagonal: t in X, s in J < X, reference row = first row of X
    for j in tl.range(0, x):
        rowj = b * T + n * C + j * BS
        offj = (rowj + t[:, None]) * H * D + dd[None, :]
        rj = tl.load(kh + offj).to(tl.float32) * tl.exp(grx[None, :] - tl.load(gh + offj))
        pidx = ((rowx + t[:, None]) * H + h) * C + j * BS + t[None, :]
        dq += tl.dot(tl.load(dpq + pidx), rj, input_precision="ieee")
        dkl += tl.dot(tl.load(dpkk + pidx), rj, input_precision="ieee")
    ex = tl.exp(gx - grx[None, :])
    dq = dq * ex
    dkl = dkl * ex
    # column role, off-diagonal: s in X, t in I > X, reference row = first row of I
    for i in tl.range(x + 1, C // BS):
        rowi = b * T + n * C + i * BS
        offi = (rowi + t[:, None]) * H * D + dd[None, :]
        gri = tl.load(gh + rowi * H * D + dd)
        ei = tl.exp(tl.load(gh + offi) - gri[None, :])
        li_q = tl.load(qh + offi).to(tl.float32) * ei
        li_k = tl.load(kh + offi).to(tl.float32) * ei
        pidx = ((rowi + t[:, None]) * H + h) * C + x * BS + t[None, :]          # [t in I, s in X]
        part = tl.dot(tl.trans(tl.load(dpq + pidx)), li_q, input_precision="ieee")
        part += tl.dot(tl.trans(tl.load(dpkk + pidx)), li_k, input_precision="ieee")
        dkr += part * tl.exp(gri[None, :] - gx)
    # diagonal block: pairwise exponentials, in BD-wide channel slices
    pidx = ((rowx + t[:, None]) * H + h) * C + x * BS + t[None, :]
    pq_d = tl.load(dpq + pidx)
    pk_d = tl.load(dpkk + pidx)
    keep = t[:, None] >= t[None, :]
    pq_d = tl.where(keep, pq_d, 0.0)
    pk_d = tl.where(t[:, None] > t[None, :], pk_d, 0.0)
    for d0 in tl.static_range(0, D, BD):
        dc = d0 + tl.arange(0, BD)
        offc = (rowx + t[:, None]) * H * D + dc[None, :]
        qc = tl.load(qh + offc).to(tl.float32)
        kc = tl.load(kh + offc).to(tl.float32)
        gc = tl.load(gh + offc)
        e3 = tl.exp(tl.minimum(gc[:, None, :] - gc[None, :, :], 0.0))      # [t, s, d]
        dq_c = tl.sum(pq_d[:, :, None] * kc[None, :, :] * e3, axis=1)     # row role: sum over s
        dkl_c = tl.sum(pk_d[:, :, None] * kc[None, :, :] * e3, axis=1)
        dkr_c = tl.sum(pq_d[:, :, None] * qc[:, None, :] * e3 + pk_d[:, :, None] * kc[:, None, :] * e3, axis=0)
        # place the slice into the full-width accumulators
        sel = (dd[None, :] >= d0) & (dd[None, :] < d0 + BD)
        dq += tl.where(sel, _widen(dq_c, d0, BS, BD, D), 0.0)
        dkl += tl.where(sel, _widen(dkl_c, d0, BS, BD, D), 0.0)
        dkr += tl.where(sel, _widen(dkr_c, d0, BS, BD, D), 0.0)
    qx = tl.load(qh + offx).to(tl.float32)
    kx = tl.load(kh + offx).to(tl.float32)
    tl.store(dqs + h * D + offx, dq)
    tl.store(dks + h * D + offx, dkl + dkr)
    tl.store(dgd + h * D + offx, qx * dq + kx * (dkl - dkr))


@triton.jit
def _widen(x, d0, BS: tl.constexpr, BD: tl.constexpr, D: tl.constexpr):
    """[BS, BD] -> [BS, D] with x in columns d0..d0+BD (zeros elsewhere), via a one-hot product."""
    src = tl.arange(0, BD)
    dst = tl.arange(0, D)
    onehot = (dst[None, :] == (src[:, None] + d0)).to(tl.float32)
    return tl.dot(x, onehot, input_precision="ieee")


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
    _gram_fwd_kernel[(T // C, B * H, C // BS)](qs, ks, gd, pkk, pq, T, H, D, C, BS, 32, num_warps=4)
    return pkk, pq


def gram_bwd(qs, ks, gd, dpkk, dpq):
    """(dqs, dks, dGd) fp32 (B, T, H, D) from the gradients of Pkk and Pq (B, T, H, 64) fp32."""
    B, T, H, D = qs.shape
    dqs = torch.empty(B, T, H, D, device=qs.device, dtype=torch.float32)
    dks, dgd = torch.empty_like(dqs), torch.empty_like(dqs)
    _gram_bwd_kernel[(T // C, B * H, C // BS)](qs, ks, gd, dpkk.contiguous(), dpq.contiguous(), dqs, dks, dgd,
                                               T, H, D, C, BS, 16, num_warps=4)
    return dqs, dks, dgd
