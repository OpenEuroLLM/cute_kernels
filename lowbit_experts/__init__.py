"""Low-bit-weight MoE expert FFN for consumer GPUs (sm_86+): weights stored as FP8 E4M3 (1 x 128 block scales,
the oellm_ambitious export's layout), int8 (per output channel) or int4 (symmetric, groups of 64 / 128, two per
byte), dequantized in registers inside the GEMM; bf16 activations, fp32 accumulation. Two kernels per expert:
gate + up as one GEMM, then down with SwiGLU applied while its input tiles load (the activation is never stored).
"""

from .kernels import LowBitWeight, expert_ffn, lowbit_linear, quantize, stack
from .reference import dequantize, expert_ffn_reference

__all__ = ["LowBitWeight", "quantize", "stack", "dequantize", "lowbit_linear", "expert_ffn", "expert_ffn_reference"]
