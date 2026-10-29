# cute_ldm numerics: the kernels against an FP64 ground truth

`python -m cute_ldm.numerics --out ldm_numerics.json` (the analysis) and
`pytest tests/test_ldm_numerics.py` (its bounds as tests). Hopper / Blackwell for the kernel parts;
`--no-cute` and the reference tests run anywhere.

## What existed before

Megatron's `tests/unit_tests/transformer/test_depth_memory_cute.py` matches the kernels to the
FP32 taped PyTorch oracle (`depth_memory_state.TorchDepthMemoryWorkspace`) for 1 to 11 edges and
512 tokens, at max-abs / max-ref of 3e-2 (reads) and 5e-2 (gradients). It also checks the read
kernel's recompute reads at 9 edges. The segment and model tests compare the kernels with
themselves, for example with and without recompute. That was all. The comparison had no ground
truth more precise than the oracle, used neither the production depth nor ill-conditioned
inputs, and never checked the gradients of `CuteRecomputeWorkspace` (the production path,
`depth_memory_unwind`).

## What is measured

- **Ground truth** (`cute_ldm/reference.py`): the recurrence as written, in plain autograd and
  any dtype, independent of the kernels and of Megatron's oracle. In FP64 it equals the closed-form
  weighted ridge solution (M = A⁻¹B, P = A⁻¹) to 4e-15, and its gradients match central
  differences (both are tests). At FP32 it matches Megatron's oracle to < 1e-6 on reads and every
  gradient.
- **Shape**: production. G 16, K 8, D 128, **97 edges** (the production model's,
  as tuned in sm90.json), window E = 4, 2 edges per recompute group. 512 tokens × 16 groups = 8192
  independent chains. Inputs are exact in every dtype (values BF16, the rest FP32), so every error belongs to
  the implementation.
- **Implementations**: `cute` (`CuteDepthMemoryWorkspace`) and `cute_recompute`
  (`CuteRecomputeWorkspace`, driven in Megatron's order: records in each group's stream, the
  collapse's backward, then the groups recomputed last first). For scale: `fp32_torch`, the FP32
  chain as Megatron's torch paths write it; `fp32_torch_sym`, the same with P symmetrized after
  each write; and `bf16_floor`, the FP64 result rounded to BF16.
- **Metric**: normwise relative error `‖x − x₆₄‖_F / ‖x₆₄‖_F` per edge (in the JSON). The tables
  show the worst edge.
- **State**: the backward's unwound M (BF16, by subtraction) and P (FP32, by addition) at every
  window start, against the FP64 forward state there. Also the conditioning of the FP64 state.

Regimes (`numerics.REGIMES`):

| regime | what | min c = λ/den | max cond(P) |
|---|---|---|---|
| nominal | production init: λ ∈ [0.97, 0.995], s ∈ [0.6, 1.4], unit keys | 0.31 | 10 |
| forget_0.85 | strong forgetting, λ ∈ [0.82, 0.88] | 0.028 | 97 |
| aligned_keys | near-collinear keys (one direction per token-group) | 0.33 | 150 |
| aligned_forget | both, λ ∈ [0.88, 0.92] | 0.039 | 1.5e3 |
| heavy_writes | write weight s² up to 16 | 0.054 | 59 |
| value_growth | values grow 1× → 13× over depth (residual stream growth) | 0.31 | 10 |

## Results (GH200, CuTe DSL 4.7.1, seed 0)

Worst edge of the normwise relative error against FP64 (`recomputed_read`: what a recomputed
layer reads; `unwound_*`: the backward's state at window starts). Seed 1 reproduces every
entry within 2 %, except the forgetting gradient (a sum over tokens), which varies by up to 2x.

| regime | impl | read | recomputed_read | khat | query | value | scale | forgetting | unwound_m | unwound_p |
|---|---|---|---|---|---|---|---|---|---|---|
| nominal | fp32_torch | 4.0e-07 |  | 7.9e-07 | 4.0e-07 | 5.7e-07 | 6.0e-07 | 1.1e-06 |  |  |
| nominal | fp32_torch_sym | 2.4e-07 |  | 2.9e-07 | 2.4e-07 | 2.3e-07 | 2.7e-07 | 5.3e-07 |  |  |
| nominal | bf16_floor | 1.7e-03 |  | 1.7e-03 | 1.7e-03 | 1.7e-03 | 1.7e-03 | 2.4e-03 |  |  |
| nominal | cute | 5.1e-03 |  | 4.8e-03 | 8.4e-03 | 4.1e-03 | 4.3e-03 | 6.6e-03 | 9.2e-03 | 2.5e-07 |
| nominal | cute_recompute | 5.1e-03 | 3.3e-02 | 7.1e-03 | 2.0e-02 | 5.4e-03 | 5.5e-03 | 8.6e-03 |  |  |
| forget_0.85 | fp32_torch | 4.8e-02 |  | 9.9e-02 | 4.8e-02 | 6.2e-02 | 7.4e-02 | 1.2e-01 |  |  |
| forget_0.85 | fp32_torch_sym | 3.2e-07 |  | 5.4e-07 | 3.2e-07 | 3.2e-07 | 5.6e-07 | 1.4e-06 |  |  |
| forget_0.85 | bf16_floor | 1.7e-03 |  | 1.7e-03 | 1.7e-03 | 1.7e-03 | 1.7e-03 | 2.5e-03 |  |  |
| forget_0.85 | cute | 3.6e-03 |  | 6.5e-03 | 1.7e-02 | 3.3e-03 | 3.8e-03 | 5.2e-03 | 1.9e-02 | 2.9e-07 |
| forget_0.85 | cute_recompute | 3.6e-03 | 6.0e-02 | 9.8e-03 | 3.4e-02 | 3.8e-03 | 4.3e-03 | 6.7e-03 |  |  |
| aligned_keys | fp32_torch | 6.6e-07 |  | 5.8e-07 | 6.5e-07 | 7.7e-07 | 9.1e-07 | 1.3e-06 |  |  |
| aligned_keys | fp32_torch_sym | 6.5e-07 |  | 5.9e-07 | 6.4e-07 | 7.3e-07 | 8.9e-07 | 1.1e-06 |  |  |
| aligned_keys | bf16_floor | 1.7e-03 |  | 1.7e-03 | 1.7e-03 | 1.7e-03 | 1.7e-03 | 2.4e-03 |  |  |
| aligned_keys | cute | 5.2e-03 |  | 5.3e-03 | 1.9e-02 | 5.2e-03 | 5.3e-03 | 5.9e-03 | 2.0e-02 | 3.2e-07 |
| aligned_keys | cute_recompute | 5.2e-03 | 4.9e-02 | 7.4e-03 | 4.2e-02 | 7.0e-03 | 7.1e-03 | 6.0e-03 |  |  |
| aligned_forget | fp32_torch | 1.2e-04 |  | 4.1e-04 | 1.2e-04 | 1.6e-04 | 6.6e-04 | 4.8e-04 |  |  |
| aligned_forget | fp32_torch_sym | 1.8e-06 |  | 2.5e-06 | 1.8e-06 | 1.8e-06 | 2.8e-06 | 5.2e-06 |  |  |
| aligned_forget | bf16_floor | 1.7e-03 |  | 1.7e-03 | 1.7e-03 | 1.7e-03 | 1.7e-03 | 2.7e-03 |  |  |
| aligned_forget | cute | 5.2e-03 |  | 6.8e-03 | 6.1e-02 | 5.1e-03 | 6.2e-03 | 7.7e-03 | 6.8e-02 | 1.1e-06 |
| aligned_forget | cute_recompute | 5.2e-03 | 1.5e-01 | 9.1e-03 | 1.2e-01 | 6.5e-03 | 7.6e-03 | 8.8e-03 |  |  |
| heavy_writes | fp32_torch | 3.6e-06 |  | 5.8e-06 | 3.6e-06 | 5.7e-06 | 5.9e-06 | 1.1e-05 |  |  |
| heavy_writes | fp32_torch_sym | 3.9e-07 |  | 6.4e-07 | 4.0e-07 | 3.9e-07 | 8.4e-07 | 1.4e-06 |  |  |
| heavy_writes | bf16_floor | 1.7e-03 |  | 1.7e-03 | 1.7e-03 | 1.7e-03 | 1.7e-03 | 2.6e-03 |  |  |
| heavy_writes | cute | 5.2e-03 |  | 5.1e-03 | 6.0e-03 | 4.0e-03 | 4.4e-03 | 5.5e-03 | 6.5e-03 | 3.3e-07 |
| heavy_writes | cute_recompute | 5.2e-03 | 2.5e-02 | 7.1e-03 | 1.3e-02 | 5.3e-03 | 5.6e-03 | 6.1e-03 |  |  |
| value_growth | fp32_torch | 4.1e-07 |  | 3.8e-06 | 4.1e-07 | 5.7e-07 | 2.6e-06 | 2.7e-06 |  |  |
| value_growth | fp32_torch_sym | 2.1e-07 |  | 4.3e-07 | 2.2e-07 | 2.3e-07 | 4.1e-07 | 4.7e-07 |  |  |
| value_growth | bf16_floor | 1.7e-03 |  | 1.7e-03 | 1.7e-03 | 1.7e-03 | 1.7e-03 | 2.5e-03 |  |  |
| value_growth | cute | 4.2e-03 |  | 8.1e-03 | 3.7e-02 | 4.1e-03 | 4.2e-03 | 4.1e-03 | 4.2e-02 | 2.5e-07 |
| value_growth | cute_recompute | 4.2e-03 | 1.7e-01 | 1.5e-02 | 9.0e-02 | 5.4e-03 | 5.4e-03 | 6.3e-03 |  |  |

## Findings

1. **Forward: stable, ~3× the BF16 floor at every depth.** Reads are 3.6e-3 to 5.2e-3 in every
   regime (floor 1.7e-3). The committed BF16 M goes from 2.4e-3 at edge 3 to 5.2e-3 at edge 96:
   M is committed once per window, so the error does not accumulate with depth. The FP32 running
   P matches FP64 to ~1e-7.
2. **Backward P unwinding (FP32, by addition): stable.** Its error is ≤ 1.1e-6 at every window
   start in every regime, including cond(P) = 1.5e3.
3. **Backward M unwinding (BF16, by subtraction): the dominant error, and it grows toward the
   early edges.** M is recovered by subtraction from a later, larger state. The absolute error
   therefore follows BF16 rounding of the *largest* M of the chain, and the relative error of a
   small early state grows. Nominal: 5e-3 deep, 9e-3 at edge 4. aligned_forget: 6.8e-2, and
   value_growth: 4.2e-2. The query gradient (M′dr) inherits it: 8e-3 nominal, 6e-2
   aligned_forget. The other gradients stay at 2–5× the floor.
4. **Recompute: recomputed reads are 6–40× less accurate than the forward's.** A recomputed layer
   reads from the running, unwound BF16 state (the read kernel), not from what the forward read.
   Against FP64: 3.3e-2 at edge 0 and ~1e-2 mid-depth in nominal, up to 0.15 (aligned_forget) and
   0.17 (value_growth) at edge 0. Under recompute the backward therefore runs on activations that
   differ from the forward's by this much in the LDM read. The read enters the layer as
   `x + read`, so relative to the layer input the error is smaller. The recompute path's query
   gradient is 2.0e-2 nominal and 0.12 aligned_forget.
5. **Megatron's FP32 torch paths are unstable under strong forgetting; the kernels are not.**
   `rls_write`, `TorchDepthMemoryWorkspace` and the unwinding workspace update
   `P ← (P − κ (kbar′P)) / λ` without symmetrizing. Rounding makes P asymmetric, and that part
   grows by 1/λ per edge, the classic RLS instability. At λ ≈ 0.85 over 97 edges the FP32 reads
   are 4.8e-2 off (gradients 0.12), worse than the BF16 kernels. The kernels keep P as a packed
   triangle, symmetric by construction. Symmetrizing after each write brings FP32 back to 3e-7
   (`fp32_torch_sym`). At λ ≈ 0.85 the unsymmetrized FP32 error passes the BF16 floor
   after ~80 edges. So an FP32 oracle without symmetrization is no reference for deep chains with
   strong forgetting.

## Possible fixes (not implemented)

- **M in FP32** for the running state the backward unwinds. The subtraction's error would then
  follow FP32 rounding of the largest state (~2^-24 instead of 2^-9), which should take the
  unwound M, the query gradient and the recomputed reads (findings 3 and 4) to the floor of the
  BF16 tapes. Not measured: it needs a kernel change. The cost is the M buffer at 4 B instead of 2 B per entry: N·G·K·D, about 2.1 GB more at
  N = 65536.
- **Symmetrize P** in Megatron's torch paths: one line each (`depth_memory.rls_write`,
  `depth_memory_state` forward and unwind).
- **Real regimes**: λ, write weights and stream growth from trained LDM checkpoints show which
  synthetic regime applies. `numerics.analyse` takes any inputs through `make_steps`, so dumped
  (khat, q, v, s, λ) can replace the synthetic ones.
