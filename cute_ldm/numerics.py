# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Numerical stability of the CuTe LDM kernels against an FP64 ground truth.

``python -m cute_ldm.numerics --out ldm_numerics.json`` runs every regime below at the
production shape (G 16, K 8, D 128, 97 edges, window 4) and prints a summary table. Per regime and
per edge it measures, against the FP64 chain of cute_ldm.reference:

- the reads and the input gradients (khat, query, value, scale, forgetting) of
  ``CuteDepthMemoryWorkspace`` (the kernels with every tape in the workspace) and of
  ``CuteRecomputeWorkspace`` (full activation recompute: tapes as stream records, the recomputed
  layers' reads from the running state by the read kernel) -- normwise relative errors;
- the same for the FP32 PyTorch chain as Megatron's torch paths write it (P updated without
  symmetrization), for the FP32 chain with a symmetrized P, and for the BF16 floor (the FP64 result rounded to BF16: what storing a read in BF16 costs alone);
- the state the backward unwinds: M (BF16, by subtraction) and P (FP32, by addition) at every
  window start against the forward's FP64 state there -- the drift of the unwinding;
- the conditioning of the FP64 state (cute_ldm.reference.diagnostics).

Inputs are exact in every dtype (values BF16, the rest FP32), so every error is the
implementation's. Tokens and groups are independent recurrences: 512 tokens x 16 groups = 8192
chains per regime.
"""

import argparse
import json

import torch

from . import reference

GRAD_NAMES = ("khat", "query", "value", "scale", "forgetting")


# -- inputs ------------------------------------------------------------------------------------
def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def make_steps(regime, edges, n, g, k, d, seed=0, device="cuda"):
    """Per edge (khat, query, value, scale, forgetting [G]) and the read cotangents, for one of
    REGIMES."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, device=device, generator=gen)

    def uni(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=device, generator=gen)

    spec = REGIMES[regime]
    lam_lo, lam_hi = spec.get("forgetting", (0.97, 0.995))
    s_lo, s_hi = spec.get("scale", (0.6, 1.4))
    base = _unit(rnd(n, g, k))   # the shared direction of aligned keys
    steps = []
    for e in range(edges):
        if "align" in spec:
            khat = _unit(base + spec["align"] * rnd(n, g, k))
        else:
            khat = _unit(rnd(n, g, k))
        query = _unit(rnd(n, g, k)) * (1 + 0.3 * rnd(n, g, 1))
        growth = 1 + spec.get("value_growth", 0.0) * e
        value = (growth * rnd(n, g, d)).to(torch.bfloat16)
        scale = uni(s_lo, s_hi, n, g)
        forgetting = uni(lam_lo, lam_hi, g)
        steps.append((khat, query, value, scale, forgetting))
    cot = [(0.1 * rnd(n, g, d)).to(torch.bfloat16) for _ in range(edges)]
    return steps, cot


REGIMES = {
    # the production init: forgetting sigmoid(logit) ~ 0.99, write weights around 1
    "nominal": dict(),
    # strong forgetting: P grows by 1/lambda per edge along directions no key excites
    "forget_0.85": dict(forgetting=(0.82, 0.88)),
    # nearly collinear keys (one direction per token-group): P ill-conditioned
    "aligned_keys": dict(align=0.1),
    # both: the unexcited directions of P grow as lambda^-e (~1e4 over 97 edges)
    "aligned_forget": dict(align=0.1, forgetting=(0.88, 0.92)),
    # heavy writes (weight s^2 up to 16): c = lambda / den small, the gain saturates
    "heavy_writes": dict(scale=(2.0, 4.0)),
    # the residual stream's growth with depth: values 1x .. 13x over 97 edges
    "value_growth": dict(value_growth=0.125),
}


# -- metrics -----------------------------------------------------------------------------------
def rel(a, ref):
    """||a - ref||_F / ||ref||_F in FP64."""
    a, ref = a.double(), ref.double()
    return float((a - ref).norm() / ref.norm().clamp_min(1e-300))


def rel_max(a, ref):
    """max |a - ref| / max |ref| (the existing kernel tests' metric)."""
    a, ref = a.double(), ref.double()
    return float((a - ref).abs().max() / ref.abs().max().clamp_min(1e-300))


def unpack_p(packed, k):
    """The packed upper-row-major FP32 triangle [N, G, >= K(K+1)/2] as [N, G, K, K]."""
    rows, cols = torch.triu_indices(k, k, device=packed.device)
    index = rows * k - rows * (rows - 1) // 2 + (cols - rows)
    full = packed.new_zeros(*packed.shape[:-1], k, k, dtype=torch.float64)
    vals = packed[..., index].double()
    full[..., rows, cols] = vals
    full[..., cols, rows] = vals
    return full


# -- the kernels -------------------------------------------------------------------------------
def run_cute(steps, cot, *, alpha, span, window_states=True):
    """CuteDepthMemoryWorkspace over the chain: reads, gradients, and the unwound state (M, P) at
    every window start s > 0 right after its window-end launch, ``{s: (M, P [N,G,K,K])}``."""
    from .workspace import CuteDepthMemoryWorkspace

    L = len(steps)
    n, g, k = steps[0][0].shape
    d = steps[0][2].shape[-1]
    ws = CuteDepthMemoryWorkspace(total_edges=L, groups=g, key_dim=k, value_dim=d, alpha=alpha,
                                  span=span, running_covariance_dtype="float32")
    ws.ensure(n, "cuda")
    reads, fwd_states = [], {}
    for e in range(L):
        reads.append(ws.forward(e, *steps[e]).float().clone())
        if window_states and ((e + 1) % ws.span == 0 or e == L - 1):
            fwd_states[e] = (ws.m[..., :d].float().clone(), unpack_p(ws.p, k))
    grads, unwound = [None] * L, {}
    for e in reversed(range(L)):
        grads[e] = [t.float().clone() for t in ws.backward(e, cot[e])]
        start = e // ws.span * ws.span
        if window_states and e == min(start + ws.span, L) - 1 and start > 0:
            unwound[start] = (ws.m[..., :d].float().clone(), unpack_p(ws.p, k))
    return dict(reads=reads, grads=grads, fwd_states=fwd_states, unwound=unwound)


def recompute_groups(edges, per_group):
    """Megatron's recompute groups: edge 0 (the embedding write) alone, then ``per_group``
    edges per chunk (one LDM site per layer, ``recompute_num_layers`` layers per chunk)."""
    groups, first = [(0, 1)], 1
    while first < edges:
        count = min(per_group, edges - first)
        groups.append((first, count))
        first += count
    return groups


def run_cute_recompute(steps, cot, *, alpha, span, per_group=2):
    """CuteRecomputeWorkspace driven in Megatron's order (depth_memory.py, DepthMemoryStepX /
    ReadFromWorkspace / StreamWrite): the forward with records into each group's stream, then the
    collapse's backward and the groups recomputed last first. Returns the forward reads, the
    recomputed reads (edge -> read the recomputed layer took) and the gradients."""
    from .workspace import CuteRecomputeWorkspace

    L = len(steps)
    n, g, k = steps[0][0].shape
    d = steps[0][2].shape[-1]
    h = g * d
    ws = CuteRecomputeWorkspace(total_edges=L, groups=g, key_dim=k, value_dim=d, alpha=alpha,
                                span=span, running_covariance_dtype="float32")
    groups = recompute_groups(L, per_group)
    ws.layout(groups, h)
    ws.ensure(n, "cuda")
    slots = ws.stream_slots(torch.bfloat16)
    streams = []   # each group's output stream [slots, n, h] (x unused: zeros)
    prev = torch.zeros(slots, n, h, dtype=torch.bfloat16, device="cuda")
    reads = [None] * L
    for first, count in groups:
        stream = prev.clone()
        for e in range(first, first + count):
            if e > 0:
                ws.ingest(e, prev)
                reads[e - 1] = ws.read(e - 1).float().clone()
            ws.forward(e, *steps[e])
            ws.put_record(e, stream[1:])
        streams.append(stream)
        prev = stream
    ws.ingest(L, streams[-1])
    reads[L - 1] = ws.read(L - 1).float().clone()
    grads, recomputed = [None] * L, {}
    grads[L - 1] = [t.float().clone() for t in ws.backward(L - 1, cot[L - 1])]
    for index in range(len(groups) - 1, 0, -1):
        first, count = groups[index]
        stream_in = streams[index - 1]
        for e in range(first, first + count):
            ws.ingest(e, stream_in)
            recomputed[e - 1] = ws.read(e - 1).float().clone()
            ws.forward(e, *steps[e])
        for e in reversed(range(first, first + count)):
            grads[e - 1] = [t.float().clone() for t in ws.backward(e - 1, cot[e - 1])]
    return dict(reads=reads, recomputed=recomputed, grads=grads)


# -- one regime --------------------------------------------------------------------------------
def analyse(regime, *, edges=97, n=512, g=16, k=8, d=128, alpha=1.0, span=4, per_group=2,
            seed=0, cute=True, recompute=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    steps, cot = make_steps(regime, edges, n, g, k, d, seed=seed)
    window_ends = [e for e in range(edges) if (e + 1) % span == 0 or e == edges - 1]
    starts = [s for s in range(span, edges, span)]
    keep = sorted(set(window_ends) | {s - 1 for s in starts})
    ref_reads, ref_grads, ref_states = reference.chain(steps, alpha, torch.float64, cot, keep)
    f32_reads, f32_grads, _ = reference.chain(steps, alpha, torch.float32, cot)
    sym_reads, sym_grads, _ = reference.chain(steps, alpha, torch.float32, cot, symmetric=True)
    out = dict(regime=regime, edges=edges, n=n, g=g, k=k, d=d, alpha=alpha, span=span,
               per_group=per_group, diagnostics=reference.diagnostics(steps, alpha), impl={})

    def score(reads, grads):
        res = dict(read=[rel(r, ref) for r, ref in zip(reads, ref_reads)],
                   read_max=[rel_max(r, ref) for r, ref in zip(reads, ref_reads)])
        for j, name in enumerate(GRAD_NAMES):
            res[name] = [rel(gr[j], ref[j]) for gr, ref in zip(grads, ref_grads)]
        return res

    out["impl"]["fp32_torch"] = score(f32_reads, f32_grads)
    out["impl"]["fp32_torch_sym"] = score(sym_reads, sym_grads)
    bf16 = [[t.to(torch.bfloat16) for t in gr] for gr in ref_grads]
    out["impl"]["bf16_floor"] = score([r.to(torch.bfloat16) for r in ref_reads], bf16)
    if cute:
        got = run_cute(steps, cot, alpha=alpha, span=span)
        res = score(got["reads"], got["grads"])
        res["fwd_m"] = {e: rel(m, ref_states[e][0]) for e, (m, _) in got["fwd_states"].items()}
        res["fwd_p"] = {e: rel(p, ref_states[e][1]) for e, (_, p) in got["fwd_states"].items()}
        res["unwound_m"] = {s: rel(m, ref_states[s - 1][0]) for s, (m, _) in got["unwound"].items()}
        res["unwound_p"] = {s: rel(p, ref_states[s - 1][1]) for s, (_, p) in got["unwound"].items()}
        out["impl"]["cute"] = res
        del got
    if recompute:
        got = run_cute_recompute(steps, cot, alpha=alpha, span=span, per_group=per_group)
        res = score(got["reads"], got["grads"])
        res["recomputed_read"] = {e: rel(r, ref_reads[e]) for e, r in got["recomputed"].items()}
        res["recomputed_vs_forward"] = {e: rel(r, got["reads"][e])
                                        for e, r in got["recomputed"].items()}
        out["impl"]["cute_recompute"] = res
        del got
    return out


# -- report ------------------------------------------------------------------------------------
def table(results):
    """Markdown: per regime and implementation the worst (over edges) normwise relative error
    of the reads and of every gradient; the state's conditioning."""
    cols = ("read", "recomputed_read") + GRAD_NAMES + ("unwound_m", "unwound_p")
    lines = ["| regime | impl | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for res in results:
        for name, impl in res["impl"].items():
            cells = []
            for c in cols:
                if c in impl and len(impl[c]):
                    vals = impl[c].values() if isinstance(impl[c], dict) else impl[c]
                    cells.append(f"{max(vals):.1e}")
                else:
                    cells.append("")
            lines.append(f"| {res['regime']} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "| regime | min c | c 1 % | max cond(P) | max eig(P) | M rms (last) | read rms (last) |",
              "|---|---|---|---|---|---|---|"]
    for res in results:
        dg = res["diagnostics"]
        lines.append(f"| {res['regime']} | {min(dg['c_min']):.1e} | {min(dg['c_p01']):.1e} | "
                     f"{max(dg['cond_p_max']):.1e} | {max(dg['eig_p_max']):.1e} | "
                     f"{dg['m_rms'][-1]:.2f} | {dg['read_rms'][-1]:.2f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    ap.add_argument("--edges", type=int, default=97)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--key-dim", type=int, default=8)
    ap.add_argument("--value-dim", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cute", action="store_true", help="references only (any GPU)")
    ap.add_argument("--out", default=None, help="JSON with every per-edge curve")
    args = ap.parse_args(argv)
    results = []
    for regime in args.regimes:
        res = analyse(regime, edges=args.edges, n=args.tokens, g=args.groups, k=args.key_dim,
                      d=args.value_dim, alpha=args.alpha, span=args.span,
                      per_group=args.per_group, seed=args.seed, cute=not args.no_cute,
                      recompute=not args.no_cute)
        results.append(res)
        print(f"[{regime}] done", flush=True)
        torch.cuda.empty_cache()
    print(table(results))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, default=str)


if __name__ == "__main__":
    main()
