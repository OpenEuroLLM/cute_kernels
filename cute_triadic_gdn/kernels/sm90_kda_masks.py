"""Triadic KDA, per chunk: the masks R and R' (Eq. 9) and the UT transform (Eq. 10) by a blocked triangular inverse,
with the channel-decayed key product Pkk (from kda_triton.gram_fwd) in place of K K^T, and the readout matrix
Pm = scale * Pq * R' (lower triangle, bf16) that the KDA forward and backward recurrences load instead of forming
Q K^T themselves.  The inverse is the GDN kernel's (sm90_joint_masks.py), unchanged.

The 16x16 inverses and first block merges use FP32. The final 32x32 merges
use TF32 operands with FP32 accumulation, matching the torch reference's
precision. Round only the completed inverse to BF16 for the recurrence.
"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu import OperandMajorMode as OMM
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from .asm import _f32, _PURE, _shfl_idx, _selp_f32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar

def fma(a,b,c):
    return cutlass.Float32(llvm.inline_asm(_f32(), [a.ir_value(), b.ir_value(), c.ir_value()],
        'fma.rn.f32 $0, $1, $2, $3;', '=f,f,f,f', **_PURE))


def _inv16_cols(col,lane):
    out=list(col)
    for row in range(1,16):
        acc=cutlass.Float32(0.)
        for k in range(row):
            pivot=_shfl_idx(out[row],k)
            acc=fma(-pivot,out[k],acc)
        out[row]=_selp_f32(acc,out[row],cutlass.Int32(lane<row))
    return out


def _mma_tf32(a, b, c):
    operands = [llvm.bitcast(ir.IntegerType.get_signless(32), x.ir_value()) for x in (*a, *b)]
    result = llvm.inline_asm(llvm.StructType.get_literal([_f32()] * 4),
        [*operands, *[x.ir_value() for x in c]],
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", **_PURE)
    return [cutlass.Float32(llvm.extractvalue(_f32(), result, [i])) for i in range(4)]


C,D,NT=64,128,128


@cute.kernel
def masks_kernel(pkk: cute.Tensor,pq: cute.Tensor,rawk2: cute.Tensor,rawq2: cute.Tensor,gc: cute.Tensor,rawbeta: cute.Tensor,
                 A: cute.Tensor,Mp: cute.Tensor,M: cute.Tensor,Pm: cute.Tensor,offsets: cute.Tensor,mapping: cute.Tensor,
                 gram: cute.TiledMma,
                 H: cutlass.Constexpr,E: cutlass.Constexpr,NC: cutlass.Constexpr,SCALE: cutlass.Constexpr,
                 PACKED: cutlass.Constexpr,NATIVE: cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    bh,n,b=bid%H,(bid//H)%NC,bid//(H*NC)
    active = cutlass.Boolean(True)
    if cutlass.const_expr(PACKED):
        active = n < offsets[cute.size(offsets)-1]
    if active:
        rows = valid_rows(n, mapping, NATIVE)
        k2 = token_chunk(rawk2,n,mapping,NATIVE)
        q2 = token_chunk(rawq2,n,mapping,NATIVE)
        beta = token_chunk(rawbeta,n,mapping,NATIVE)
        lane,wid=tid%32,tid//32
        smem=utils.SmemAllocator()
        tmp=smem.allocate_tensor(cutlass.Float32,cute.make_layout((32,32),stride=(36,1)),byte_alignment=16)
        backing=smem.allocate_tensor(cutlass.BFloat16,cute.make_layout(C*(C+4)*2),byte_alignment=128)
        sf=cute.make_tensor(cute.recast_ptr(backing.iterator,dtype=cutlass.Float32),cute.make_layout((C,C),stride=(C+4,1)))
        sg=smem.allocate_tensor(cutlass.Float32,cute.make_layout((E,C),stride=(C+32//E,1)),byte_alignment=16)
        sk2=smem.allocate_tensor(cutlass.Float32,cute.make_layout((E,C),stride=(C+32//E,1)),byte_alignment=16)
        sq=smem.allocate_tensor(cutlass.Float32,cute.make_layout((E,C),stride=(C+32//E,1)),byte_alignment=16)
        sb=smem.allocate_tensor(cutlass.Float32,cute.make_layout(C),byte_alignment=16)
        for j in cutlass.range_constexpr((E*C+NT-1)//NT):
            pos=j*NT+tid
            if pos<E*C:
                r,e=pos//E,pos%E
                sg[e,r]=gc[b,n,r,bh,e]
                sk2[e,r]=masked_scalar(k2,(b,n,r,bh,e),r,rows)
                sq[e,r]=masked_scalar(q2,(b,n,r,bh,e),r,rows)
        if tid<C:
            sb[tid]=masked_scalar(beta,(b,n,tid,bh),tid,rows)
        cute.arch.fence_proxy('async.shared',space='cta')
        cute.arch.barrier()
        th=gram.get_slice(tid)
        cp=th.partition_C(cute.make_identity_tensor((C,C)))
        cute.arch.barrier()
        outm,outmp=cute.make_rmem_tensor((2,),cutlass.BFloat16),cute.make_rmem_tensor((2,),cutlass.BFloat16)
        outf=cute.make_rmem_tensor((2,),cutlass.Float32)
        outpm=cute.make_rmem_tensor((2,),cutlass.BFloat16)
        for j in cutlass.range_constexpr(C*C//(NT*2)):
            for z in cutlass.range_constexpr(2):
                i=2*j+z
                r,c=cp[i][0],cp[i][1]
                m,mp=cutlass.Float32(0.),cutlass.Float32(0.)
                if c<=r:
                    for e in cutlass.range_constexpr(E):
                        decay=cute.math.exp(sg[e,r]-sg[e,c],fastmath=True)
                        m=m+sk2[e,r]*sk2[e,c]*decay
                        mp=mp+sq[e,r]*sk2[e,c]*decay
                val=cutlass.Float32(0.)
                pmv=cutlass.Float32(0.)
                if c<=r:
                    pmv=SCALE*pq[b,n,r,bh,c]*mp
                if c<r:
                    val=sb[r]*pkk[b,n,r,bh,c]*m
                elif c==r:
                    val=cutlass.Float32(1.)
                outf[z]=val
                if c>=r:
                    m=cutlass.Float32(0.)
                outm[z]=cutlass.BFloat16(m)
                outmp[z]=cutlass.BFloat16(mp)
                outpm[z]=cutlass.BFloat16(pmv)
            r,c=cp[2*j][0],cp[2*j][1]
            dst=cute.make_tensor((sf.iterator+sf.layout((r,c))).align(8),cute.make_layout((2,)))
            dst.store(outf.load())
            dst=cute.make_tensor((M.iterator+M.layout((b,n,r,bh,c))).align(4),cute.make_layout((2,)))
            dst.store(outm.load())
            dst=cute.make_tensor((Mp.iterator+Mp.layout((b,n,r,bh,c))).align(4),cute.make_layout((2,)))
            dst.store(outmp.load())
            dst=cute.make_tensor((Pm.iterator+Pm.layout((b,n,r,bh,c))).align(4),cute.make_layout((2,)))
            dst.store(outpm.load())
        cute.arch.barrier()
        cols=[]
        for r in cutlass.range_constexpr(16):
            val=cutlass.Float32(0.)
            if lane<16:
                val=sf[wid*16+r,wid*16+lane]
            cols.append(val)
        cols=_inv16_cols(cols,lane)
        for r in cutlass.range_constexpr(16):
            if lane<16:
                sf[wid*16+r,wid*16+lane]=cols[r]
        cute.arch.barrier()
        # Inv([[L,0],[X,R]]) = [[Linv,0],[-Rinv X Linv,Rinv]].
        # Keep first-level products in FP32; match TF32 operands at the final level.
        for size in cutlass.range_constexpr(16,17,16):
            for j in cutlass.range_constexpr((C//(2*size))*size*size//NT):
                pos=j*NT+tid
                block,r,c=pos//(size*size),(pos//size)%size,pos%size
                offset=block*2*size
                value=cutlass.Float32(0.)
                for t in cutlass.range_constexpr(size):
                    left,right=sf[offset+size+r,offset+size+t],sf[offset+size+t,offset+c]
                    value=fma(-left,right,value)
                tmp[block*size+r,c]=value
            cute.arch.barrier()
            for j in cutlass.range_constexpr((C//(2*size))*size*size//NT):
                pos=j*NT+tid
                block,r,c=pos//(size*size),(pos//size)%size,pos%size
                offset=block*2*size
                value=cutlass.Float32(0.)
                for t in cutlass.range_constexpr(size):
                    left,right=tmp[block*size+r,t],sf[offset+t,offset+c]
                    value=fma(left,right,value)
                sf[offset+size+r,offset+c]=value
            cute.arch.barrier()
        # The last block merge uses the same TF32 MMA reduction as the torch reference.
        # Scalar FMA with truncated operands rounds at different points in the sum.
        mr, mc = (wid % 2) * 16, (wid // 2) * 16
        row, col = lane // 4, lane % 4
        for phase in cutlass.range_constexpr(2):
            cc0 = [cutlass.Float32(0.) for _ in range(4)]
            cc1 = [cutlass.Float32(0.) for _ in range(4)]
            for kk in cutlass.range_constexpr(0,32,8):
                if cutlass.const_expr(phase == 0):
                    aa = [sf[32+mr+row,32+kk+col], sf[32+mr+row+8,32+kk+col],
                          sf[32+mr+row,32+kk+col+4], sf[32+mr+row+8,32+kk+col+4]]
                    bb0 = [-sf[32+kk+col,mc+row], -sf[32+kk+col+4,mc+row]]
                    bb1 = [-sf[32+kk+col,mc+row+8], -sf[32+kk+col+4,mc+row+8]]
                else:
                    aa = [tmp[mr+row,kk+col], tmp[mr+row+8,kk+col],
                          tmp[mr+row,kk+col+4], tmp[mr+row+8,kk+col+4]]
                    bb0 = [sf[kk+col,mc+row], sf[kk+col+4,mc+row]]
                    bb1 = [sf[kk+col,mc+row+8], sf[kk+col+4,mc+row+8]]
                cc0 = _mma_tf32(aa,bb0,cc0)
                cc1 = _mma_tf32(aa,bb1,cc1)
            for j in cutlass.range_constexpr(4):
                rr, c0 = mr+row+(j//2)*8, mc+col*2+j%2
                if cutlass.const_expr(phase == 0):
                    tmp[rr,c0],tmp[rr,c0+8]=cc0[j],cc1[j]
                else:
                    sf[32+rr,c0],sf[32+rr,c0+8]=cc0[j],cc1[j]
            cute.arch.barrier()
        pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
        for j in cutlass.range_constexpr(C*C//(NT*2)):
            i=2*j
            r,c=cp[i][0],cp[i][1]
            pack[0],pack[1]=cutlass.BFloat16(sf[r,c]),cutlass.BFloat16(sf[r,c+1])
            dst=cute.make_tensor((A.iterator+A.layout((b,n,r,bh,c))).align(4),cute.make_layout((2,)))
            dst.store(pack.load())


@cute.jit
def launch_masks(pkk: cute.Tensor,pq: cute.Tensor,k2: cute.Tensor,q2: cute.Tensor,gc: cute.Tensor,beta: cute.Tensor,
                 A: cute.Tensor,Mp: cute.Tensor,M: cute.Tensor,Pm: cute.Tensor,offsets: cute.Tensor,mapping: cute.Tensor,
                 B: cutlass.Constexpr,H: cutlass.Constexpr,E: cutlass.Constexpr,NC: cutlass.Constexpr,SCALE: cutlass.Constexpr,
                 PACKED: cutlass.Constexpr,NATIVE: cutlass.Constexpr,stream: cuda.CUstream):
    bf,ff=cutlass.BFloat16,cutlass.Float32
    # only its C-fragment partition is used: the per-thread (row, column) ownership of the epilogue
    gram=sm90.make_trivial_tiled_mma(bf,bf,OMM.K,OMM.K,ff,(1,1,1),(C,C))
    masks_kernel(pkk,pq,k2,q2,gc,beta,A,Mp,M,Pm,offsets,mapping,gram,H,E,NC,SCALE,PACKED,NATIVE).launch(
        grid=(B*NC*H,1,1),block=(NT,1,1),stream=stream)


_compiled={}


def kda_masks(pkk,pq,k2,q2,gc,beta,scale):
    """A, R' (Mp), R (M), Pm: (B, T, H, 64) bf16 each.  pkk, pq (B, T, H, 64) fp32 from kda_triton.gram_fwd."""
    B,T,H,_=pkk.shape
    E,NC=k2.shape[-1],T//C
    assert T%C==0
    A,Mp,M,Pm=(torch.empty((B,T,H,C),dtype=torch.bfloat16,device=pkk.device) for _ in range(4))
    args=[from_dlpack(t.detach().view(B,NC,C,*t.shape[2:]),assumed_align=16)
          for t in (pkk,pq,k2,q2,gc,beta,A,Mp,M,Pm)]
    args+=[args[0],args[0]]          # offsets, mapping: unused (no packed documents)
    stream=cuda.CUstream(torch.cuda.current_stream(pkk.device).cuda_stream)
    key=(B,T,H,E,float(scale))
    if key not in _compiled:
        _compiled[key]=cute.compile(launch_masks,*args,B,H,E,NC,float(scale),False,False,stream)
    _compiled[key](*args,stream)
    return A,Mp,M,Pm
