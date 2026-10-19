"""Autograd wrappers over tile_kernels.mhc's read (pre_apply_mix) and write (post) kernels."""

import torch

from tile_kernels.mhc import (
    mhc_post_bwd,
    mhc_post_fwd,
    mhc_pre_apply_mix_bwd,
    mhc_pre_apply_mix_fwd,
)


class _Read(torch.autograd.Function):
    """``[s, b, n, C]`` bf16 streams, ``[s, b, n]`` read weights -> ``[s, b, C]`` bf16."""

    @staticmethod
    def forward(ctx, streams, h_pre):
        s, b, n, c = streams.shape
        x = streams.contiguous().view(s * b, n, c)
        mix = h_pre.float().contiguous().view(s * b, n)
        out = torch.empty(s * b, c, dtype=torch.bfloat16, device=streams.device)
        mhc_pre_apply_mix_fwd(x, mix, out)
        ctx.save_for_backward(x, mix)
        ctx.shape, ctx.h_pre_dtype = (s, b, n, c), h_pre.dtype
        return out.view(s, b, c)

    @staticmethod
    def backward(ctx, grad_out):
        x, mix = ctx.saved_tensors
        s, b, n, c = ctx.shape
        x_grad = torch.zeros_like(x)  # the kernel accumulates into it
        mix_grad = mhc_pre_apply_mix_bwd(grad_out.contiguous().view(s * b, c), x, mix, x_grad)
        return x_grad.view(s, b, n, c), mix_grad.view(s, b, n).to(ctx.h_pre_dtype)


class _Write(torch.autograd.Function):
    """``streams' = h_res @ streams + h_post^T f``: ``[s, b, n, C]`` bf16 streams, ``[s, b, C]`` bf16
    sublayer output, ``[s, b, n]`` / ``[s, b, n, n]`` (target, source) coefficients."""

    @staticmethod
    def forward(ctx, streams, sublayer_out, h_post, h_res):
        s, b, n, c = streams.shape
        residual = streams.contiguous()
        x = sublayer_out.contiguous()
        post = h_post.float().contiguous().view(s, b, n, 1)
        comb = h_res.float().transpose(-1, -2).contiguous()  # [source, target]
        out = mhc_post_fwd(x, residual, post, comb)
        ctx.save_for_backward(x, residual, post, comb)
        ctx.dtypes = (h_post.dtype, h_res.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, residual, post, comb = ctx.saved_tensors
        d_x, d_residual, d_post, d_comb = mhc_post_bwd(x, residual, post, comb, grad_out)
        post_dtype, res_dtype = ctx.dtypes
        return (
            d_residual,
            d_x,
            d_post.view(post.shape[:-1]).to(post_dtype),
            d_comb.transpose(-1, -2).to(res_dtype),
        )


def mhc_read(streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Megatron's ``_mhc_read`` through TileKernels."""
    return _Read.apply(streams, h_pre)


def mhc_write(streams, sublayer_out, h_post, h_res) -> torch.Tensor:
    """Megatron's ``_mhc_write`` through TileKernels."""
    return _Write.apply(streams, sublayer_out, h_post, h_res)
