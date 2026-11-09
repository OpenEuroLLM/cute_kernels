"""Quantization error of real expert weights (a sample of the oellm_ambitious FP8 export): the FP8 weights decoded
to fp32 are the reference; int8 per channel and int4 (symmetric RTN / asymmetric RTN / symmetric with a per-group
clipping-ratio search, groups of 64 / 128) against it -- relative weight error and the relative error of the
expert's output on Gaussian inputs (the experts read RMS-normalised hidden states).

    python -m lowbit_experts.quant_error <sample.bin> --out quant_error.json

The sample: a JSON-header container (uint64 header length, JSON {name: {dtype, shape, offset, size}}, raw bytes)
of single experts' gate_up_proj / down_proj E4M3 bytes and their 1 x 128 fp32 scales.
"""

from __future__ import annotations

import argparse
import json
import struct

import torch
import torch.nn.functional as F

from lowbit_experts import quantize
from lowbit_experts.reference import dequantize


def load_sample(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    items = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in items.items():
        if not isinstance(meta, dict):
            out[name] = meta
            continue
        buf = bytearray(raw[base + meta["offset"]: base + meta["offset"] + meta["size"]])
        dt = {"U8": torch.uint8, "F32": torch.float32}[meta["dtype"]]
        out[name] = torch.frombuffer(buf, dtype=dt).view(meta["shape"]).clone()
    return out


def decode_fp8(q, s, block=128):
    w = q.view(torch.float8_e4m3fn).float()
    n, k = w.shape
    return (w.view(n, k // block, block) * s.float()[..., None]).view(n, k)


def asym_int4(w, group):
    """Asymmetric RTN int4 (scale + zero point per group), dequantized (analysis only; the kernel is symmetric)."""
    n, k = w.shape
    g = w.view(-1, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    s = ((hi - lo) / 15.0).clamp(min=1e-12)
    q = ((g - lo) / s).round().clamp(0, 15)
    return (q * s + lo).view(n, k)


def ffn(x, wgu, wd):
    gu = x @ wgu.t()
    i = gu.shape[1] // 2
    return (F.silu(gu[:, :i]) * gu[:, i:]) @ wd.t()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sample")
    p.add_argument("--out", default="quant_error.json")
    args = p.parse_args()
    smp = load_sample(args.sample)
    experts = sorted({k.rsplit("_gate_up_proj", 1)[0] for k in smp if k.endswith("_gate_up_proj")})
    torch.manual_seed(0)
    x = torch.randn(512, 2048, device="cuda")
    variants = {
        "int8_channel": lambda w: dequantize(quantize(w, "int8")),
        "int4_sym_g128": lambda w: dequantize(quantize(w, "int4", 128)),
        "int4_sym_g64": lambda w: dequantize(quantize(w, "int4", 64)),
        "int4_sym_search_g128": lambda w: dequantize(quantize(w, "int4", 128, search=True)),
        "int4_sym_search_g64": lambda w: dequantize(quantize(w, "int4", 64, search=True)),
        "int4_asym_g128": lambda w: asym_int4(w, 128),
        "int4_asym_g64": lambda w: asym_int4(w, 64),
    }
    rows = []
    for ex in experts:
        layer = int(ex.split("_")[0][1:])
        e = int(ex.split("_")[1][1:])
        wgu = decode_fp8(smp[f"{ex}_gate_up_proj"], smp[f"{ex}_gate_up_proj_scale"]).cuda()
        wd = decode_fp8(smp[f"{ex}_down_proj"], smp[f"{ex}_down_proj_scale"]).cuda()
        y = ffn(x, wgu, wd)
        row = {"expert": ex, "layer": layer, "e": e, "runaway": e == smp.get(f"L{layer}_runaway"),
               "absmax_gu": float(wgu.abs().max()), "absmax_d": float(wd.abs().max()),
               "kurtosis_d": float(((wd - wd.mean()) ** 4).mean() / wd.var() ** 2)}
        for name, fn in variants.items():
            qgu, qd = fn(wgu), fn(wd)
            row[name] = {"w_gu": rel(qgu, wgu), "w_d": rel(qd, wd), "out": rel(ffn(x, qgu, qd), y)}
        rows.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()})
                          for k, v in row.items()}), flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
