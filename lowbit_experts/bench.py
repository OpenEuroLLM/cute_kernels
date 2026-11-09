"""Time one SwiGLU expert (v4 shape: hidden 2048, expert ffn 768) per weight format and token count against
torch baselines and the weight-bytes / memory-bandwidth bound.

    python -m lowbit_experts.bench --out bench.json [--grouped-gemm <path to oellm_kernels.py>]

Two timings per case: eager (CUDA events around N back-to-back calls, launch overhead included) and CUDA-graph
replay of N calls (kernel time; what a decode engine with graphs pays).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import torch
import torch.nn.functional as F

from lowbit_experts import expert_ffn, quantize, stack
from lowbit_experts.reference import dequantize

H, I = 2048, 768


def timed(fn, iters=200, graph=False):
    """Mean us per call of fn(), eager or replayed from a CUDA graph of `iters` calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3 / iters
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def mem_bandwidth_gbs() -> float:
    """Device-to-device copy bandwidth (read + write), GB/s."""
    a = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    us = timed(lambda: b.copy_(a), iters=20)
    return 2 * a.numel() / (us * 1e-6) / 1e9


def load_grouped(path):
    spec = importlib.util.spec_from_file_location("oellm_kernels_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oellm_kernels_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="bench.json")
    p.add_argument("--grouped-gemm", default="", help="oellm_kernels.py of the HF export (its FP8 grouped GEMM)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--grouped", action="store_true", help="per-expert time inside grouped launches of 5 / 48 experts")
    args = p.parse_args()
    if args.grouped:
        return grouped_main(args)
    torch.manual_seed(0)
    dev = "cuda"
    wgu = torch.randn(2 * I, H, device=dev) * 0.02
    wd = torch.randn(H, I, device=dev) * 0.02
    bw = mem_bandwidth_gbs()
    kern = load_grouped(args.grouped_gemm) if args.grouped_gemm else None
    formats = [("bf16", 128), ("fp8", 128), ("int8", 128), ("int4", 128), ("int4", 64)]
    weights = {f"{f}_g{g}": (quantize(wgu, f, g), quantize(wd, f, g)) for f, g in formats}
    rows = []
    for m in (1, 4, 16, 64):
        x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
        # torch bf16 baseline: two matmuls (gate | up fused) + silu * up + down
        wgu_b, wd_b = wgu.to(torch.bfloat16), wd.to(torch.bfloat16)

        def torch_bf16():
            gu = x @ wgu_b.t()
            return (F.silu(gu[:, :I]) * gu[:, I:]) @ wd_b.t()

        rows.append({"impl": "torch_bf16", "m": m, "bytes": (wgu_b.numel() + wd_b.numel()) * 2,
                     "eager_us": timed(torch_bf16, args.iters), "graph_us": timed(torch_bf16, args.iters, True)})
        # torch dequant baselines (dequantize every call, then the bf16 path)
        for name in ("fp8_g128", "int8_g128", "int4_g128"):
            a, b = weights[name]

            def torch_deq(a=a, b=b):
                gu = x @ dequantize(a).to(torch.bfloat16).t()
                return (F.silu(gu[:, :I]) * gu[:, I:]) @ dequantize(b).to(torch.bfloat16).t()

            rows.append({"impl": f"torch_dequant_{name}", "m": m, "bytes": a.nbytes() + b.nbytes(),
                         "eager_us": timed(torch_deq, 50), "graph_us": timed(torch_deq, 50, True)})
        # the fused kernels
        for name, (a, b) in weights.items():
            rows.append({"impl": f"fused_{name}", "m": m, "bytes": a.nbytes() + b.nbytes(),
                         "eager_us": timed(lambda a=a, b=b: expert_ffn(x, a, b), args.iters),
                         "graph_us": timed(lambda a=a, b=b: expert_ffn(x, a, b), args.iters, True)})
        # the HF export's grouped FP8 GEMM (in-kernel dequant, software E4M3 on sm_86), one expert
        if kern is not None:
            a, b = weights["fp8_g128"]
            idx = torch.zeros(m, 1, dtype=torch.int64, device=dev)
            bm = 16 if m <= 16 else (32 if m <= 32 else 64)
            sched = kern.ExpertSchedule(idx, 1, bm)
            wa, sa, wb_, sb = a.data[None], a.scale[None], b.data[None], b.scale[None]

            def grouped():
                gu = kern.grouped_gemm(x, sched, wa, sa, gather=True)
                act = (F.silu(gu[:, :I].float()) * gu[:, I:].float()).to(torch.bfloat16)
                return kern.grouped_gemm(act, sched, wb_, sb, gather=False)

            rows.append({"impl": "hf_export_grouped_fp8", "m": m, "bytes": a.nbytes() + b.nbytes(),
                         "eager_us": timed(grouped, args.iters), "graph_us": timed(grouped, args.iters, True)})
    for r in rows:
        r["bound_us"] = r["bytes"] / (bw * 1e9) * 1e6
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump({"device": torch.cuda.get_device_name(), "mem_bandwidth_gbs": bw, "rows": rows},
              open(args.out, "w"), indent=1)


def grouped_main(args):
    """All E experts of a layer in one launch per GEMM (as a MoE decode step runs them): graph-replay time per
    expert vs the per-expert bandwidth bound."""
    torch.manual_seed(0)
    bw = mem_bandwidth_gbs()
    rows = []
    for fmt, g in [("bf16", 128), ("fp8", 128), ("int8", 128), ("int4", 128), ("int4", 64)]:
        for e_n in (1, 5, 48):
            ws = [(quantize(torch.randn(2 * I, H, device="cuda") * 0.02, fmt, g),
                   quantize(torch.randn(H, I, device="cuda") * 0.02, fmt, g)) for _ in range(e_n)]
            a, b = stack([w[0] for w in ws]), stack([w[1] for w in ws])
            for m in (1, 16):
                x = torch.randn(e_n, m, H, device="cuda", dtype=torch.bfloat16)
                us = timed(lambda: expert_ffn(x, a, b), 100, True)
                r = {"impl": f"fused_{fmt}_g{g}", "experts": e_n, "m": m, "graph_us": us, "us_per_expert": us / e_n,
                     "bound_us_per_expert": (a.nbytes() + b.nbytes()) / e_n / (bw * 1e9) * 1e6}
                rows.append(r)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump({"device": torch.cuda.get_device_name(), "mem_bandwidth_gbs": bw, "rows": rows},
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
