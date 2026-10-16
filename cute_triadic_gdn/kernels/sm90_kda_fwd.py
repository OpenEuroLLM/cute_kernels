"""Triadic KDA forward recurrence for E <= 4: the GDN kernel (sm90_joint_fwd.py) with untied channel-decayed
operands.  The value warpgroup reads the state with kd and the output warpgroup with qd (Q tile); the state
warpgroup writes it with the update key ku (its own tile) and decays every row (e, d) by exp(g2_L,e) lam_d; the
intra-chunk readout matrix Pm = scale * Pq * R' arrives precomputed (kda_masks) instead of being formed from
Q K^T.  Stores the state at the start of every chunk in BF16 for the backward."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import warpgroup, cpasync
from cutlass.cute.nvgpu import OperandMajorMode as OMM
from cutlass.cute.runtime import from_dlpack
from .sm90_joint_packed import token_chunk, valid_rows, masked_scalar, zero_tail, store_pair, keep_singleton_tma_axis
from .sm90_joint_fwd import _store_mma_bf16

from cutlass._mlir.dialects import llvm
from .asm import _f32, _PURE


def fma(a,b,c):
    return cutlass.Float32(llvm.inline_asm(_f32(), [a.ir_value(), b.ir_value(), c.ir_value()],
        'fma.rn.f32 $0, $1, $2, $3;', '=f,f,f,f', **_PURE))


C = 64          # chunk
WG = 128        # threads per warpgroup



@cute.kernel
def fwd_kernel(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor,
                  gO: cute.Tensor, gHs: cute.Tensor, gWo: cute.Tensor, gVdo: cute.Tensor, gLam: cute.Tensor,
                  tmaQ: cute.CopyAtom, rawQt: cute.Tensor,
                  tmaK: cute.CopyAtom, rawKt: cute.Tensor,
                  tmaKu: cute.CopyAtom, rawKut: cute.Tensor,
                  tmaV: cute.CopyAtom, rawVt: cute.Tensor,
                  tmaA: cute.CopyAtom, tAt: cute.Tensor,
                  tmaM: cute.CopyAtom, tMt: cute.Tensor,
                  tmaH: cute.CopyAtom, tHt: cute.Tensor,
                  gOffsets: cute.Tensor, mapping: cute.Tensor,
                  mmaP: cute.TiledMma, mmaO: cute.TiledMma, mmaS: cute.TiledMma,
                  lQ: cute.ComposedLayout, lK: cute.ComposedLayout, lKa: cute.ComposedLayout,
                  lKq: cute.ComposedLayout, lV: cute.ComposedLayout, lH: cute.ComposedLayout,
                  lVn: cute.ComposedLayout, lP: cute.ComposedLayout, lA: cute.ComposedLayout,
                  B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr,
                  KD: cutlass.Constexpr, VC: cutlass.Constexpr, NC: cutlass.Constexpr, SAVE: cutlass.Constexpr,
                  SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NATIVE: cutlass.Constexpr):
    tid,_,_=cute.arch.thread_idx()
    bid,_,_=cute.arch.block_idx()
    lane=tid%WG
    group=cute.arch.make_warp_uniform(tid//WG)
    warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
    NV=128//VC
    vb,bh,b=bid%NV,(bid//NV)%H,bid//(NV*H)
    if cutlass.const_expr(DOCS and NATIVE):
        b = cutlass.Int32(mapping[2, b])      # documents longest first (see metadata_kernel)
    first, count = cutlass.Int32(0), cutlass.Int32(NC)
    if cutlass.const_expr(DOCS):
        first = cutlass.Int32(gOffsets[b])
        count = cutlass.Int32(gOffsets[b+1]) - first
        b = cutlass.Int32(0)
    smem=utils.SmemAllocator()
    def alloc(layout):
        return smem.allocate_tensor(cutlass.BFloat16,layout.outer,128,swizzle=layout.inner)
    def view(t,layout):
        return cute.make_tensor(cute.recast_ptr(t.iterator,layout.inner,dtype=cutlass.BFloat16),layout.outer)
    def fp(shape,stride=None):
        return smem.allocate_tensor(cutlass.Float32,cute.make_layout(shape,stride=stride),byte_alignment=16)
    sQ,sK,sKu,sV,sA,sM=alloc(lQ),alloc(lK),alloc(lK),alloc(lV),alloc(lA),alloc(lA)
    sH,sVn,sWt,sVd,sP=alloc(lH),alloc(lVn),alloc(lV),alloc(lV),alloc(lP)
    sKa,sKq,sQo=view(sKu,lKa),view(sK,lKq),view(sQ,lKq)
    sa,sr,sc=[fp((2,E,C),(E*(C+32//E),C+32//E,1)) for _ in range(3)]
    sb,sd=fp((2,C),(C,1)),fp((2,E),(E,1))
    slam=fp((2,KD),(KD,1))
    def bar(n=1):
        return smem.allocate_tensor(cutlass.Int64,cute.make_layout(n),byte_alignment=8)
    ready,free=bar(2),bar(2)
    tailReady=bar(2) if cutlass.const_expr(NATIVE == 1) else ready
    hReady,hFree,vnReady,vdReady,vdFree=bar(),bar(),bar(),bar(),bar()
    if tid<32:
        with cute.arch.elect_one():
            for st in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(ready.iterator+st,WG+1)
                if cutlass.const_expr(NATIVE == 1):
                    cute.arch.mbarrier_init(tailReady.iterator+st,WG)
                cute.arch.mbarrier_init(free.iterator+st,3*WG)
            cute.arch.mbarrier_init(hReady.iterator,WG)
            cute.arch.mbarrier_init(hFree.iterator,2*WG)
            cute.arch.mbarrier_init(vnReady.iterator,WG)
            cute.arch.mbarrier_init(vdReady.iterator,WG)
            cute.arch.mbarrier_init(vdFree.iterator,WG)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    def tma_part(atom,ten,sm,mma,tile,is_a):
        gt=cute.local_tile(ten,tile,(None,None,None,None,None))
        th=mma.get_slice(0)
        part=th.partition_A(gt) if is_a else th.partition_B(gt)
        return cpasync.tma_partition(atom,0,cute.make_layout(1),cute.group_modes(sm,0,2),cute.group_modes(part,0,3))
    if cutlass.const_expr(NATIVE):
        start = mapping[0, first]
        tQt = cute.domain_offset((start,0,0,0,0), rawQt)
        tKt = cute.domain_offset((start,0,0,0,0), rawKt)
        tVt = cute.domain_offset((0,start,0,0,0), rawVt)
    else:
        tQt, tKt, tVt = rawQt, rawKt, rawVt
    tQs,tQg=tma_part(tmaQ,tQt,sQ,mmaP,(C,KD),True)
    tKs,tKg=tma_part(tmaK,tKt,sK,mmaP,(C,KD),False)
    tKus,tKug=tma_part(tmaKu,rawKut,sKu,mmaP,(C,KD),False)
    tVs,tVg=tma_part(tmaV,tVt,sV,mmaO,(VC,C),False)
    tAs,tAg=tma_part(tmaA,tAt,sA,mmaO,(C,C),True)
    tMs,tMg=tma_part(tmaM,tMt,sM,mmaO,(C,C),True)
    if cutlass.const_expr(SAVE):
        tHs,tHg=tma_part(tmaH,tHt,sH,mmaO,(VC,KD),False)
    tu,tp,ts=mmaO.get_slice(lane),mmaP.get_slice(lane),mmaS.get_slice(lane)
    cu=tu.partition_C(cute.make_identity_tensor((C,VC)))
    cp=tp.partition_C(cute.make_identity_tensor((C,C)))
    cs=ts.partition_C(cute.make_identity_tensor((KD,VC)))
    NU,NS,NP=C*VC//WG,KD*VC//WG,C*C//WG
    def sync(group):
        cute.arch.barrier(barrier_id=group+1,number_of_threads=WG)
    def gemm_async(mma,acc,fa,fb,add=False):
        warpgroup.fence()
        for k in cutlass.range(cute.size(fa,mode=[2]),unroll_full=True):
            mma.set(warpgroup.Field.ACCUMULATE,cutlass.Boolean(add or k!=0))
            cute.gemm(mma,acc,fa[None,None,k],fb[None,None,k],acc)
        warpgroup.commit_group()
    if group==3:
        cute.arch.warpgroup_reg_dealloc(64)
        for step in cutlass.range(count):
            n = first + step
            st=step%2
            if step>=2:
                cute.arch.mbarrier_wait(free.iterator+st,((step-2)//2)%2)
            rows = valid_rows(n, mapping, NATIVE)
            k2c = token_chunk(gK2, n, mapping, NATIVE)
            qq = token_chunk(gQ2, n, mapping, NATIVE)
            qb = token_chunk(gBt, n, mapping, NATIVE)
            if warp==12:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ready.iterator+st,6*C*KD+2*C*VC+4*C*C)
                if cutlass.const_expr(NATIVE):
                    cute.copy(tmaQ,tQg[(None,step,0,b,0,bh)],tQs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaK,tKg[(None,step,0,b,0,bh)],tKs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaV,tVg[(None,vb,step,b,0,bh)],tVs[(None,st)],tma_bar_ptr=ready.iterator+st)
                else:
                    cute.copy(tmaQ,tQg[(None,0,0,b,n,bh)],tQs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaK,tKg[(None,0,0,b,n,bh)],tKs[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaKu,tKug[(None,0,0,b,n,bh)],tKus[(None,st)],tma_bar_ptr=ready.iterator+st)
                    cute.copy(tmaV,tVg[(None,vb,0,b,n,bh)],tVs[(None,st)],tma_bar_ptr=ready.iterator+st)
                cute.copy(tmaA,tAg[(None,0,0,b,n,bh)],tAs[(None,st)],tma_bar_ptr=ready.iterator+st)
                cute.copy(tmaM,tMg[(None,0,0,b,n,bh)],tMs[(None,st)],tma_bar_ptr=ready.iterator+st)
            for j in cutlass.range_constexpr((E*C+WG-1)//WG):
                pos=j*WG+lane
                if pos<E*C:
                    e,r=pos%E,pos//E
                    g= gGc[b,n,r,bh,e]
                    gL=gGc[b,n,C-1,bh,e]
                    weight=masked_scalar(k2c,(b,n,r,bh,e),r,rows)
                    eg=cute.math.exp(g,fastmath=True)
                    sa[st,e,r]=weight*eg
                    sr[st,e,r]=weight*cute.math.exp(gL-g,fastmath=True)
                    sc[st,e,r]=SCALE*masked_scalar(qq,(b,n,r,bh,e),r,rows)*eg
            if lane<C:
                sb[st,lane]=masked_scalar(qb,(b,n,lane,bh),lane,rows)
            if lane<E:
                sd[st,lane]=cute.math.exp(gGc[b,n,C-1,bh,lane],fastmath=True)
            slam[st,lane]=gLam[b,n,bh,lane]          # WG == KD lanes: one channel each
            cute.arch.mbarrier_arrive(ready.iterator+st)
            if cutlass.const_expr(NATIVE):
                if rows < C:
                    cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
                    zero_tail(sQ,rows,lane,WG,KD,st,False)
                    zero_tail(sK,rows,lane,WG,KD,st,False)
                    zero_tail(sV,rows,lane,WG,VC,st,True)
                    cute.arch.fence_proxy('async.shared',space='cta')
                    sync(group)
                    cute.arch.mbarrier_arrive(tailReady.iterator+st)
    elif group==0:
        cute.arch.warpgroup_reg_alloc(160)
        states=[mmaS.make_fragment_C(mmaS.partition_shape_C((KD,VC))) for _ in range(E)]
        for e in cutlass.range_constexpr(E):
            states[e].fill(0.)
        for step in cutlass.range(count):
            n = first + step
            st=step%2
            cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
            if cutlass.const_expr(NATIVE):
                if valid_rows(n,mapping,NATIVE) < C:
                    cute.arch.mbarrier_wait(tailReady.iterator+st,0)
            if step>0:
                cute.arch.mbarrier_wait(hFree.iterator,(step-1)%2)
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for e in cutlass.range_constexpr(E):
                _store_mma_bf16(states[e],sH,lH,KD,VC,e,lane)
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            if cutlass.const_expr(SAVE):
                if warp==0:
                    for e in cutlass.range_constexpr(E):
                        cute.copy(tmaH,tHs[(None,e)],tHg[(None,vb,e,b,n,bh)])
                    cute.arch.cp_async_bulk_commit_group()
            cute.arch.mbarrier_arrive(hReady.iterator)
            cute.arch.mbarrier_wait(vnReady.iterator,step%2)
            fa=mmaS.make_fragment_A(ts.partition_A(sKa[None,None,st]))
            for e in cutlass.range_constexpr(E):
                decay=sd[st,e]
                for i in cutlass.range_constexpr(NS):
                    states[e][i]=states[e][i]*(decay*slam[st,cs[i][0]])
                fb=mmaS.make_fragment_B(ts.partition_B(sVn[None,None,e]))
                gemm_async(mmaS,states[e],fa,fb,True)
            warpgroup.wait_group(0)
            if cutlass.const_expr(SAVE):
                if warp==0:
                    cute.arch.cp_async_bulk_wait_group(0,read=True)
            cute.arch.mbarrier_arrive(free.iterator+st)
        if cutlass.const_expr(SAVE):
            if warp==0:
                cute.arch.cp_async_bulk_wait_group(0)
    elif group==1:
        cute.arch.warpgroup_reg_alloc(144)
        for step in cutlass.range(count):
            n = first + step
            st=step%2
            cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
            if cutlass.const_expr(NATIVE):
                if valid_rows(n,mapping,NATIVE) < C:
                    cute.arch.mbarrier_wait(tailReady.iterator+st,0)
            cute.arch.mbarrier_wait(hReady.iterator,step%2)
            accs=[mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC))) for _ in range(E)]
            fa=mmaO.make_fragment_A(tu.partition_A(sKq[None,None,st]))
            for e in cutlass.range_constexpr(E):
                fb=mmaO.make_fragment_B(tu.partition_B(sH[None,None,e]))
                gemm_async(mmaO,accs[e],fa,fb)
            warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(hFree.iterator)
            gates=cute.make_rmem_tensor((E,2),cutlass.Float32)
            for e in cutlass.range_constexpr(E):
                gates[e,0],gates[e,1]=sa[st,e,cu[0][0]],sa[st,e,cu[2][0]]
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for j in cutlass.range_constexpr(NU//2):
                i=2*j
                r,c=cu[i][0],cu[i][1]
                v0,v1=cutlass.Float32(sV[c,r,st]),cutlass.Float32(sV[c+1,r,st])
                for e in cutlass.range_constexpr(E):
                    weight=gates[e,j%2]
                    v0,v1=fma(-weight,accs[e][i],v0),fma(-weight,accs[e][i+1],v1)
                if cutlass.const_expr(SAVE):
                    pack[0],pack[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                    dst=cute.make_tensor((gWo.iterator+gWo.layout((b,n,r,bh,vb*VC+c))).align(4),cute.make_layout((2,)))
                    dst.store(pack.load())
                pack[0],pack[1]=cutlass.BFloat16(v0),cutlass.BFloat16(v1)
                accs[0][i],accs[0][i+1]=v0,v1
            _store_mma_bf16(accs[0],sWt,lV,C,VC,0,lane)
            # Match the reference's BF16 factorization: (A diag(beta)) @ W.
            # Scaling W instead is algebraically equivalent but rounds differently
            # at every recurrent layer. Keep the saved, global A unscaled.
            ab=mmaP.make_fragment_C(mmaP.partition_shape_C((C,C)))
            for i in cutlass.range_constexpr(NP):
                r,c=cp[i][0],cp[i][1]
                ab[i]=cutlass.Float32(sA[r,c,st])*sb[st,c]
            _store_mma_bf16(ab,sA,lA,C,C,st,lane,True)
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            fa=mmaO.make_fragment_A(tu.partition_A(sA[None,None,st]))
            fb=mmaO.make_fragment_B(tu.partition_B(sWt[None,None,0]))
            gemm_async(mmaO,accs[0],fa,fb)
            warpgroup.wait_group(0)
            if step>0:
                cute.arch.mbarrier_wait(vdFree.iterator,(step-1)%2)
            for j in cutlass.range_constexpr(NU//2):
                i=2*j
                r,c=cu[i][0],cu[i][1]
                pack[0],pack[1]=cutlass.BFloat16(accs[0][i]),cutlass.BFloat16(accs[0][i+1])

                if cutlass.const_expr(SAVE):
                    dst=cute.make_tensor((gVdo.iterator+gVdo.layout((b,n,r,bh,vb*VC+c))).align(4),cute.make_layout((2,)))
                    dst.store(pack.load())
            _store_mma_bf16(accs[0],sVd,lV,C,VC,0,lane)
            cute.arch.fence_proxy('async.shared',space='cta')
            cute.arch.mbarrier_arrive(vdReady.iterator)
            vn=mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC)))
            for e in cutlass.range_constexpr(E):
                weight0=sr[st,e,cu[0][0]]
                weight1=sr[st,e,cu[2][0]]
                for j in cutlass.range_constexpr(NU//2):
                    i=2*j
                    weight=weight0 if cutlass.const_expr(j%2==0) else weight1
                    vn[i],vn[i+1]=accs[0][i]*weight,accs[0][i+1]*weight
                _store_mma_bf16(vn,sVn,lVn,C,VC,e,lane)
            cute.arch.fence_proxy('async.shared',space='cta')
            cute.arch.mbarrier_arrive(vnReady.iterator)
            cute.arch.mbarrier_arrive(free.iterator+st)
    else:
        cute.arch.warpgroup_reg_alloc(144)
        for step in cutlass.range(count):
            n = first + step
            rows_out = valid_rows(n,mapping,NATIVE)
            out_chunk = token_chunk(gO,n,mapping,NATIVE)
            st=step%2
            cute.arch.mbarrier_wait(ready.iterator+st,(step//2)%2)
            if cutlass.const_expr(NATIVE):
                if valid_rows(n,mapping,NATIVE) < C:
                    cute.arch.mbarrier_wait(tailReady.iterator+st,0)
            ap=mmaP.make_fragment_C(mmaP.partition_shape_C((C,C)))
            for i in cutlass.range_constexpr(NP):
                r,c=cp[i][0],cp[i][1]
                ap[i]=cutlass.Float32(sM[r,c,st])     # Pm: masked and scaled already
            _store_mma_bf16(ap,sP,lP,C,C,0,lane,True)
            cute.arch.mbarrier_wait(hReady.iterator,step%2)
            accs=[mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC))) for _ in range(E)]
            ao=mmaO.make_fragment_C(mmaO.partition_shape_C((C,VC)))
            fa=mmaO.make_fragment_A(tu.partition_A(sQo[None,None,st]))
            for e in cutlass.range_constexpr(E):
                fb=mmaO.make_fragment_B(tu.partition_B(sH[None,None,e]))
                gemm_async(mmaO,accs[e],fa,fb)
            warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(hFree.iterator)
            for i in cutlass.range_constexpr(NU):
                val=cutlass.Float32(0.)
                # The first readout addition contracts c0*u0 into c1*u1.
                for e in cutlass.range_constexpr(E):
                    sl=(1-e if e<2 else e) if E>1 else e
                    val=fma(sc[st,sl,cu[i][0]],accs[sl][i],val)
                ao[i]=val
            cute.arch.fence_proxy('async.shared',space='cta')
            sync(group)
            cute.arch.mbarrier_arrive(free.iterator+st)
            cute.arch.mbarrier_wait(vdReady.iterator,step%2)
            fa=mmaO.make_fragment_A(tu.partition_A(sP[None,None,0]))
            fb=mmaO.make_fragment_B(tu.partition_B(sVd[None,None,0]))
            gemm_async(mmaO,ao,fa,fb,True)
            warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(vdFree.iterator)
            pack=cute.make_rmem_tensor((2,),cutlass.BFloat16)
            for j in cutlass.range_constexpr(NU//2):
                i=2*j
                pack[0],pack[1]=cutlass.BFloat16(ao[i]),cutlass.BFloat16(ao[i+1])
                store_pair(out_chunk,b,n,cu[i][0],bh,vb*VC+cu[i][1],pack.load(),rows_out)


@cute.jit
def launch_fwd(gGc: cute.Tensor, gK2: cute.Tensor, gQ2: cute.Tensor, gBt: cute.Tensor,
                        gO: cute.Tensor, gHs: cute.Tensor, gWo: cute.Tensor, gVdo: cute.Tensor, gLam: cute.Tensor, gQA: cute.Tensor, gKB: cute.Tensor, gKuB: cute.Tensor, gVB: cute.Tensor,
                        gAA: cute.Tensor, gMA: cute.Tensor, gHT: cute.Tensor, gOffsets: cute.Tensor, mapping: cute.Tensor,
                        B: cutlass.Constexpr, H: cutlass.Constexpr, E: cutlass.Constexpr,
                        KD: cutlass.Constexpr, VC: cutlass.Constexpr, NC: cutlass.Constexpr, SAVE: cutlass.Constexpr,
                        SCALE: cutlass.Constexpr, DOCS: cutlass.Constexpr, NBLK: cutlass.Int32, NATIVE: cutlass.Constexpr, stream: cuda.CUstream):
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    one = (1, 1, 1)
    mmaP = sm90.make_trivial_tiled_mma(bf16, bf16, OMM.K, OMM.K, f32, one, (C, C))      # Q K^T
    mmaO = sm90.make_trivial_tiled_mma(bf16, bf16, OMM.K, OMM.MN, f32, one, (C, VC))    # K@S, A@W, P@Vd, Q@S
    mmaS = sm90.make_trivial_tiled_mma(bf16, bf16, OMM.MN, OMM.MN, f32, one, (C, VC))   # K^T@vn (M=KD via M-iters)
    lQ = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, C, KD), bf16, 2)
    lK = sm90.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (C, C, KD), bf16, 2)
    lKa = sm90.make_smem_layout_a(LayoutEnum.COL_MAJOR, (KD, VC, C), bf16, 2)
    lKq = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, VC, KD), bf16, 2)
    lV = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, C), bf16, 2)
    lH = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, KD), bf16, E)
    lVn = sm90.make_smem_layout_b(LayoutEnum.COL_MAJOR, (C, VC, C), bf16, E)
    lP = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, VC, C), bf16, 1)
    lA = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, VC, C), bf16, 2)
    op = cpasync.CopyBulkTensorTileG2SOp()
    tmaQ, tQt = cpasync.make_tiled_tma_atom(op, gQA, cute.slice_(lQ, (None, None, 0)), (C, KD), 1)
    tmaK, tKt = cpasync.make_tiled_tma_atom(op, gKB, cute.slice_(lK, (None, None, 0)), (C, KD), 1)
    tmaKu, tKut = cpasync.make_tiled_tma_atom(op, gKuB, cute.slice_(lK, (None, None, 0)), (C, KD), 1)
    tmaV, tVt = cpasync.make_tiled_tma_atom(op, gVB, cute.slice_(lV, (None, None, 0)), (VC, C), 1)
    tmaA, tAt = cpasync.make_tiled_tma_atom(op, gAA, cute.slice_(lA, (None, None, 0)), (C, C), 1)
    tmaM, tMt = cpasync.make_tiled_tma_atom(op, gMA, cute.slice_(lA, (None, None, 0)), (C, C), 1)
    tmaH,tHt=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),gHT,
        cute.slice_(lH,(None,None,0)),(VC,KD),1)
    fwd_kernel(gGc, gK2, gQ2, gBt, gO, gHs, gWo, gVdo, gLam, tmaQ, tQt, tmaK, tKt, tmaKu, tKut, tmaV, tVt, tmaA, tAt, tmaM, tMt, tmaH, tHt, gOffsets, mapping,
                  mmaP, mmaO, mmaS, lQ, lK, lKa, lKq, lV, lH, lVn, lP, lA,
                  B, H, E, KD, VC, NC, SAVE, SCALE, DOCS, NATIVE).launch(
        grid=(NBLK * H * (128 // VC), 1, 1), block=(4 * WG, 1, 1), stream=stream, min_blocks_per_mp=1)


_compiled = {}


def kda_fwd(qd, kd, ku, v, k2, q2, gc, beta, A, Pm, lam, scale=None, value_cols=None, save=False):
    """Return o, or (o, states, W, Vd) when save=True.

    qd, kd, ku, v (B,T,H,128) bf16; k2, q2, gc (B,T,H,E) f32 (gc = chunk-local cumsum of g2); beta (B,T,H) f32;
    A, Pm (B,T,H,64) bf16 from kda_masks; lam (B,NC,H,128) f32.  No packed documents."""
    B, Tn, H, KD = qd.shape
    DV = v.shape[-1]
    E = k2.shape[-1]
    assert KD == 128 and DV == 128 and Tn % C == 0
    assert E in (1, 2, 4), "E >= 8: kda_fwd_split"
    VC = 128//E if value_cols is None else value_cols
    VC = min(VC, 32)
    assert VC in (16, 32) and E*VC <= 128
    NC = Tn // C
    scale = KD ** -0.5 if scale is None else float(scale)
    dev = qd.device
    o = torch.empty(B, Tn, H, DV, device=dev, dtype=torch.bfloat16)
    hs = torch.empty((B,NC,H,E,KD,DV),device=dev,dtype=qd.dtype) if save else o
    wo,vdo = (torch.empty((B,Tn,H,DV),device=dev,dtype=o.dtype) for _ in range(2)) if save else (o,o)
    def matrix_view(t):
        return t.view(B,NC,C,H,C).permute(2,4,0,1,3)
    def raw(t):
        return t.view(B,NC,C,*t.shape[2:])
    q5, k5, ku5, v5 = (raw(t) for t in (qd,kd,ku,v))
    args = (gc.view(B, NC, C, H, E), raw(k2), raw(q2),
            raw(beta), raw(o),
            hs, raw(wo) if save else raw(o), raw(vdo) if save else raw(o), lam,
            q5.permute(2, 4, 0, 1, 3), k5.permute(2, 4, 0, 1, 3), ku5.permute(2, 4, 0, 1, 3),
            v5.permute(4, 2, 0, 1, 3),
            matrix_view(A), matrix_view(Pm),
            hs.view(B,NC,H,E*KD,DV).permute(4,3,0,1,2) if save else q5.permute(4,2,0,1,3))
    ta = [from_dlpack(t.detach(), assumed_align=16) for t in args]
    ta.append(ta[0])                  # offsets: unused
    ta.append(ta[0])                  # mapping: unused
    key = (B, Tn, H, E, scale, VC, bool(save), NC)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    if key not in _compiled:
        _compiled[key] = cute.compile(launch_fwd, *ta, B, H, E, KD, VC, NC, bool(save), float(scale), False, B, 0, stream)
    _compiled[key](*ta, B, stream)
    return (o,hs,wo,vdo) if save else o
