# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""The LDM recurrence in plain PyTorch autograd, in any dtype: the ground truth (FP64) of the
numerics analysis (cute_ldm.numerics) and the FP32 reference it is compared with.

Independent of the kernels and of Megatron's taped oracle (depth_memory_state.py): no tapes, no
windows, no unwinding -- the step as written, with autograd for the backward. Every edge ``e``
writes and then reads::

    kbar = s khat,  vbar = s v,  P kbar,  den = lambda + kbar' P kbar,  kappa = P kbar / den,
    u = vbar - M' kbar,  M <- M + kappa u',  P <- (P - kappa (P kbar)') / lambda,  read = M' q

starting from M = 0, P = alpha I. No matmul / einsum: broadcast products and sums only, so TF32
(the NGC containers enable it for cuBLAS) never enters an FP32 run. Tokens and groups are
independent recurrences, so long inputs run in token chunks.
"""

import torch


def _step(m, p, khat, query, value, scale, lam, symmetric=False):
    kbar = khat * scale.unsqueeze(-1)
    vbar = value * scale.unsqueeze(-1)
    pk = (p * kbar.unsqueeze(-2)).sum(-1)                    # [n, g, k]  P kbar
    den = lam + (kbar * pk).sum(-1)                          # [n, g]
    kappa = pk / den.unsqueeze(-1)
    u = vbar - (m * kbar.unsqueeze(-1)).sum(-2)              # [n, g, d]
    m = m + kappa.unsqueeze(-1) * u.unsqueeze(-2)
    p = (p - kappa.unsqueeze(-1) * pk.unsqueeze(-2)) / lam.unsqueeze(-1).unsqueeze(-1)
    if symmetric:  # the kernels keep P as a packed triangle: symmetric by construction
        p = 0.5 * (p + p.transpose(-1, -2))
    read = (m * query.unsqueeze(-1)).sum(-2)                 # [n, g, d]
    return m, p, read, den


def _chunk(steps, lo, hi, alpha, dtype, grad, states, symmetric):
    """The chain on tokens [lo, hi): reads, and (grad) the leaves; ``states``: edges whose
    (M, P) after the write to keep (detached)."""
    leaves, reads, kept = [], [], {}
    khat0 = steps[0][0]
    n, g, k = hi - lo, khat0.shape[1], khat0.shape[2]
    d = steps[0][2].shape[-1]
    dev = khat0.device
    m = torch.zeros(n, g, k, d, dtype=dtype, device=dev)
    p = (torch.eye(k, dtype=dtype, device=dev) * alpha).expand(n, g, k, k)
    for e, (khat, query, value, scale, lam) in enumerate(steps):
        xs = [t[lo:hi].to(dtype) for t in (khat, query, value, scale)]
        lam = lam.to(dtype)
        if lam.dim() == 1:
            lam = lam.expand(n, g)
        else:
            lam = lam[lo:hi]
        if grad:
            xs = [t.detach().requires_grad_(True) for t in xs]
            lam = lam.detach().requires_grad_(True)
            leaves.append((*xs, lam))
        m, p, read, _ = _step(m, p, *xs, lam, symmetric)
        reads.append(read)
        if e in states:
            kept[e] = (m.detach().clone(), p.detach().clone())
    return leaves, reads, kept


def chain(steps, alpha=1.0, dtype=torch.float64, read_grads=None, states=(), chunk=128,
          symmetric=False):
    """The reads of every edge and, with ``read_grads`` (one cotangent per edge), the gradients
    of (khat, query, value, scale, forgetting) per edge -- forgetting summed over tokens to
    ``[G]`` when it was given per group.

    ``steps``: per edge ``(khat [N,G,K], query [N,G,K], value [N,G,D], scale [N,G],
    forgetting [G] or [N,G])``. ``symmetric``: symmetrize P after every write. Without it the
    rounding makes P asymmetric, and with strong forgetting the asymmetric part grows by
    1 / lambda per edge (the classic RLS instability; see cute_ldm/NUMERICS.md). ``states``:
    edges whose state after the write to return, ``{e: (M [N,G,K,D], P [N,G,K,K])}``. Returns ``(reads, grads, states)`` in ``dtype``."""
    n = steps[0][0].shape[0]
    states = set(states)
    reads = [[] for _ in steps]
    grads = [[[] for _ in range(5)] for _ in steps] if read_grads is not None else None
    lam_grads = [0 for _ in steps]
    kept = {}
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        with torch.enable_grad() if read_grads is not None else torch.no_grad():
            leaves, out, part = _chunk(steps, lo, hi, alpha, dtype, read_grads is not None, states,
                                     symmetric)
            for e, r in enumerate(out):
                reads[e].append(r.detach())
            for e, (m, p) in part.items():
                kept.setdefault(e, ([], []))
                kept[e][0].append(m)
                kept[e][1].append(p)
            if read_grads is None:
                continue
            cot = [dr[lo:hi].to(dtype) for dr in read_grads]
            flat = [t for edge in leaves for t in edge]
            got = torch.autograd.grad(out, flat, cot, allow_unused=True)
        for e in range(len(steps)):
            for j in range(5):
                gj = got[5 * e + j]
                if gj is None:
                    gj = torch.zeros_like(flat[5 * e + j])
                if j == 4 and steps[e][4].dim() == 1:
                    lam_grads[e] = lam_grads[e] + gj.sum(0)
                else:
                    grads[e][j].append(gj)
    reads = [torch.cat(r) for r in reads]
    if grads is not None:
        grads = [[torch.cat(g[j]) if j < 4 or steps[e][4].dim() == 2 else lam_grads[e]
                  for j in range(5)] for e, g in enumerate(grads)]
    kept = {e: (torch.cat(ms), torch.cat(ps)) for e, (ms, ps) in kept.items()}
    return reads, grads, kept


@torch.no_grad()
def diagnostics(steps, alpha=1.0, chunk=256):
    """Per edge, in FP64: the conditioning of the state the kernels round --
    ``c = lambda / den = 1 - kappa . kbar`` (the share of P left along the written key; the
    closed-form unwind divides by it), the condition number and the largest eigenvalue of P,
    and the RMS of M and of the read."""
    n = steps[0][0].shape[0]
    out = {key: [] for key in ("c_min", "c_p01", "cond_p_max", "cond_p_med", "eig_p_max",
                               "m_rms", "read_rms")}
    rows = {key: [[] for _ in steps] for key in ("c", "cond", "eigmax", "m2", "r2")}
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        khat0 = steps[0][0]
        g, k, d = khat0.shape[1], khat0.shape[2], steps[0][2].shape[-1]
        m = torch.zeros(hi - lo, g, k, d, dtype=torch.float64, device=khat0.device)
        p = (torch.eye(k, dtype=torch.float64, device=khat0.device) * alpha).expand(hi - lo, g, k, k)
        for e, (khat, query, value, scale, lam) in enumerate(steps):
            lam = lam.double()
            lam = lam.expand(hi - lo, g) if lam.dim() == 1 else lam[lo:hi]
            m, p, read, den = _step(m, p, khat[lo:hi].double(), query[lo:hi].double(),
                                    value[lo:hi].double(), scale[lo:hi].double(), lam)
            eig = torch.linalg.eigvalsh(p)
            rows["c"][e].append((lam / den).flatten())
            rows["cond"][e].append((eig[..., -1] / eig[..., 0]).flatten())
            rows["eigmax"][e].append(eig[..., -1].flatten())
            rows["m2"][e].append(m.pow(2).mean().reshape(1) * m.numel())
            rows["r2"][e].append(read.pow(2).mean().reshape(1) * read.numel())
    total_m = n * steps[0][0].shape[1] * steps[0][0].shape[2] * steps[0][2].shape[-1]
    total_r = n * steps[0][0].shape[1] * steps[0][2].shape[-1]
    for e in range(len(steps)):
        c = torch.cat(rows["c"][e])
        cond = torch.cat(rows["cond"][e])
        out["c_min"].append(float(c.min()))
        out["c_p01"].append(float(torch.quantile(c.float(), 0.01)))
        out["cond_p_max"].append(float(cond.max()))
        out["cond_p_med"].append(float(cond.median()))
        out["eig_p_max"].append(float(torch.cat(rows["eigmax"][e]).max()))
        out["m_rms"].append(float((torch.cat(rows["m2"][e]).sum() / total_m).sqrt()))
        out["read_rms"].append(float((torch.cat(rows["r2"][e]).sum() / total_r).sqrt()))
    return out
