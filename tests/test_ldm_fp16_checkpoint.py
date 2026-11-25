"""FP16 storage under BF16 training (``grad_scaling``) and M checkpoints (``checkpoint_interval``)
of CuteRecomputeWorkspace, driven in Megatron's order with a BF16 stream, against the FP64 chain of
cute_ldm.reference.

The production failure mode is a burst of large writes deep in the stack (MoE sublayers 55-71 of
the 97-edge production model): the backward unwinds M by subtraction from the larger state after
the burst, so every state before it loses precision. Here the values of those edges are scaled by
``burst``, and the cotangents are activation-gradient sized (~1e-7, BF16), which FP16 cannot hold
unscaled. `pytest tests/test_ldm_fp16_checkpoint.py`; `python tests/test_ldm_fp16_checkpoint.py`
prints the error table of every configuration.
"""
import pytest
import torch

from cute_ldm import numerics, reference
from cute_ldm.workspace import CuteRecomputeWorkspace

L, N, G, K, D, SPAN, PER_GROUP = 97, 512, 16, 8, 128, 4, 2
BAND = set(range(55, 72, 2))
BEFORE = [e for e in range(L) if e < min(BAND)]
COT_SCALE = 1e-7


def _hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (9, 10)


needs_gpu = pytest.mark.skipif(not _hopper(), reason="CuTe kernels need Hopper / Blackwell")


def make_inputs(burst, cot_scale=COT_SCALE, seed=0, device="cuda"):
    """The nominal regime with the band's values scaled by ``burst``; values and cotangents BF16,
    as Megatron hands them to the workspace under BF16 training."""
    steps, cot = numerics.make_steps("nominal", L, N, G, K, D, seed=seed, device=device)
    steps = [(khat, query, (value.float() * (burst if e in BAND else 1.0)).to(torch.bfloat16),
              scale, lam) for e, (khat, query, value, scale, lam) in enumerate(steps)]
    cot = [(c.float() * (cot_scale / 0.1)).to(torch.bfloat16) for c in cot]
    return steps, cot


def run(steps, cot, *, storage="bfloat16", grad_scaling=False, interval=0, offload=True,
        prefetch=False, stream_dtype=torch.bfloat16, workspace=None):
    """The workspace as Megatron drives it (DepthMemoryMixer / DepthMemoryStepX /
    ReadFromWorkspace / StreamWrite): the forward with each edge's record put into its group's
    stream, the final read, then the groups recomputed last first, each group's edges backward in
    reverse. Returns the forward reads, the recomputed reads and the input gradients (FP32)."""
    ws = workspace or CuteRecomputeWorkspace(
        total_edges=L, groups=G, key_dim=K, value_dim=D, span=SPAN,
        running_covariance_dtype="float32", storage_dtype=storage, grad_scaling=grad_scaling,
        checkpoint_interval=interval, checkpoint_offload=offload, checkpoint_prefetch=prefetch)
    groups = numerics.recompute_groups(L, PER_GROUP)
    if workspace is None:
        ws.layout(groups, G * D)
        ws.ensure(N, "cuda")
    slots = ws.stream_slots(stream_dtype)
    prev = torch.zeros(slots, N, G * D, dtype=stream_dtype, device="cuda")
    streams, reads = [], [None] * L
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
        for e in range(first, first + count):
            ws.ingest(e, streams[index - 1])
            recomputed[e - 1] = ws.read(e - 1).float().clone()
            ws.forward(e, *steps[e])
        for e in reversed(range(first, first + count)):
            grads[e - 1] = [t.float().clone() for t in ws.backward(e - 1, cot[e - 1])]
    torch.cuda.synchronize()
    return dict(reads=reads, recomputed=recomputed, grads=grads, workspace=ws)


def errors(got, ref_reads, ref_grads, edges=BEFORE):
    """Worst edge in ``edges`` of the normwise relative error of the recomputed read and of every
    input gradient, against FP64."""
    out = dict(recomputed_read=max(numerics.rel(got["recomputed"][e], ref_reads[e])
                                   for e in edges if e in got["recomputed"]))
    for j, name in enumerate(numerics.GRAD_NAMES):
        out[name] = max(numerics.rel(got["grads"][e][j], ref_grads[e][j]) for e in edges)
    return out


def ground_truth(steps, cot):
    reads, grads, _ = reference.chain(steps, 1.0, torch.float64, cot)
    return reads, grads


CONFIGS = {
    "bf16": dict(storage="bfloat16"),
    "bf16+ckpt16": dict(storage="bfloat16", interval=16),
    "fp16": dict(storage="float16", grad_scaling=True),
    "fp16+ckpt32": dict(storage="float16", grad_scaling=True, interval=32),
    "fp16+ckpt16": dict(storage="float16", grad_scaling=True, interval=16),
    "fp16+ckpt8": dict(storage="float16", grad_scaling=True, interval=8),
}


# -- layout (CPU) ----------------------------------------------------------------------------------
def test_checkpoint_boundaries():
    ws = CuteRecomputeWorkspace(total_edges=L, groups=G, key_dim=K, value_dim=D, span=SPAN,
                                checkpoint_interval=16)
    ws.layout(numerics.recompute_groups(L, PER_GROUP), G * D)
    assert ws.checkpoints == [16, 32, 48, 64, 80, 96]
    for bad in (6, 10):
        ws = CuteRecomputeWorkspace(total_edges=L, groups=G, key_dim=K, value_dim=D, span=SPAN,
                                    checkpoint_interval=bad)
        with pytest.raises(ValueError, match="multiple of the span"):
            ws.layout(numerics.recompute_groups(L, PER_GROUP), G * D)


def test_grad_scaling_needs_fp16():
    with pytest.raises(ValueError, match="grad_scaling"):
        CuteRecomputeWorkspace(total_edges=L, groups=G, key_dim=K, value_dim=D, grad_scaling=True)


# -- kernels -------------------------------------------------------------------------------------
@needs_gpu
def test_records_carry_fp16_bits():
    """FP16 u in a BF16 stream is stored as its bits: the run equals one with an FP16 stream (where
    the records hold u as is), bit for bit."""
    steps, cot = make_inputs(burst=30.0)
    a = run(steps, cot, storage="float16", grad_scaling=True, stream_dtype=torch.bfloat16)
    b = run(steps, cot, storage="float16", grad_scaling=True, stream_dtype=torch.float16)
    for e in range(L):
        for x, y in zip(a["grads"][e], b["grads"][e]):
            assert torch.equal(x, y), e
    for e, r in a["recomputed"].items():
        assert torch.equal(r, b["recomputed"][e]), e


@needs_gpu
def test_grad_scaling_is_exact():
    """The backward is linear in the cotangents and the scale a power of two: cotangents 2^-20
    smaller give input gradients exactly 2^-20 smaller."""
    steps, cot = make_inputs(burst=1.0, cot_scale=1e-1)
    small = [(c.float() * 2.0 ** -20).to(torch.bfloat16) for c in cot]
    a = run(steps, cot, storage="float16", grad_scaling=True)
    b = run(steps, small, storage="float16", grad_scaling=True)
    for e in range(L):
        for x, y in zip(a["grads"][e], b["grads"][e]):
            assert torch.equal(x * 2.0 ** -20, y), e
    assert int(a["workspace"].grad_nonfinite) == 0


@needs_gpu
def test_unscaled_fp16_underflows():
    """Why grad_scaling exists: FP16 tapes with activation-gradient-sized cotangents and no scale
    lose the gradients; the scaled run keeps them at the FP16 floor."""
    steps, cot = make_inputs(burst=1.0)
    ref_reads, ref_grads = ground_truth(steps, cot)
    scaled = errors(run(steps, cot, storage="float16", grad_scaling=True), ref_reads, ref_grads)
    unscaled = errors(run(steps, cot, storage="float16"), ref_reads, ref_grads)
    assert unscaled["value"] > 0.1
    assert scaled["value"] < 1e-2


@needs_gpu
def test_checkpoint_offload_and_prefetch_equal_device():
    """Host checkpoints, with and without the staging prefetch, equal device checkpoints bit for
    bit, over two micro-batches in a row."""
    steps, cot = make_inputs(burst=85.0)
    runs = {}
    for name, kw in dict(device=dict(offload=False), host=dict(offload=True),
                         prefetch=dict(offload=True, prefetch=True)).items():
        first = run(steps, cot, storage="float16", grad_scaling=True, interval=16, **kw)
        runs[name] = [first, run(steps, cot, workspace=first["workspace"])]
    for name in ("host", "prefetch"):
        for got, ref in zip(runs[name], runs["device"]):
            for e in range(L):
                for x, y in zip(got["grads"][e], ref["grads"][e]):
                    assert torch.equal(x, y), (name, e)


@needs_gpu
def test_grad_scale_adapts_to_larger_cotangents():
    """Cotangents far above the first one's saturate instead of overflowing; the gradients stay
    finite, and the next backward scales from the larger amax."""
    steps, cot = make_inputs(burst=1.0)
    cot = [c if e > 40 else (c.float() * 2.0 ** 16).to(torch.bfloat16) for e, c in enumerate(cot)]
    got = run(steps, cot, storage="float16", grad_scaling=True)
    ws = got["workspace"]
    assert all(torch.isfinite(t).all() for g in got["grads"] for t in g)
    first = float(ws.grad_scale)
    amax = max(float(c.float().abs().max()) for c in cot)
    assert float(ws.grad_amax_prev) >= amax
    run(steps, cot, workspace=ws)
    assert float(ws.grad_scale) * amax <= 2.0 ** 5 < first * amax


@needs_gpu
@pytest.mark.parametrize("burst", [1.0, 85.0])
def test_fp16_checkpoints_against_fp64(burst):
    """At the run's burst (~85x) the production path (BF16, no checkpoints) recomputes the reads
    before the burst with errors of several percent; FP16 with checkpoints every 16 edges brings
    them and the query gradient down by more than 8x."""
    steps, cot = make_inputs(burst)
    ref_reads, ref_grads = ground_truth(steps, cot)
    base = errors(run(steps, cot, **CONFIGS["bf16"]), ref_reads, ref_grads)
    best = errors(run(steps, cot, **CONFIGS["fp16+ckpt16"]), ref_reads, ref_grads)
    for name in ("recomputed_read", "query"):
        assert best[name] * 8 < base[name], (name, best[name], base[name])
    for name in ("khat", "value", "scale"):
        assert best[name] < 2e-3, (name, best[name])


def table(bursts=(1.0, 10.0, 30.0, 85.0, 300.0)):
    cols = ("recomputed_read",) + numerics.GRAD_NAMES
    print("worst edge before the band (< 55), normwise relative error against FP64")
    print("| burst | config | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for burst in bursts:
        steps, cot = make_inputs(burst)
        ref_reads, ref_grads = ground_truth(steps, cot)
        for name, config in CONFIGS.items():
            got = errors(run(steps, cot, **config), ref_reads, ref_grads)
            print(f"| {burst:g} | {name} | " + " | ".join(f"{got[c]:.1e}" for c in cols) + " |",
                  flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    table()
