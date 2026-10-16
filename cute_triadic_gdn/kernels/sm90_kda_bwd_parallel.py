"""Triadic KDA backward, second stage: every chunk in parallel (from sm90_joint_bwd_parallel.py).

The matrix kernel forms dbeta, the derivatives of the masks R and R' and, instead of folding the derivatives of
the channel-decayed products into dq / dk (which the GDN kernel does through Q K^T and K K^T), stores them:
dPq = scale * R' * (dO Vd^T) (lower) and dPkk = dL * beta * R (strictly lower), for kda_triton.gram_bwd.  The R / R'
reduction into dk2, dq2 and the slice decay gradient is the GDN kernel (mask_grad_kernel), unchanged.
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
from .sm90_joint_packed import masked_scalar
from .sm90_joint_bwd_parallel import launch_mask_grad

C, D, NT = 64, 128, 128


@cute.kernel
def matrix_kernel(pq: cute.Tensor, pkk: cute.Tensor, do: cute.Tensor,
                  a: cute.Tensor, mp: cute.Tensor, m: cute.Tensor, beta: cute.Tensor,
                  wv: cute.Tensor, vd: cute.Tensor, dvd: cute.Tensor,
                  dpq: cute.Tensor, dpkk: cute.Tensor, db: cute.Tensor,
                  dmp: cute.Tensor, dm: cute.Tensor,
                  gram: cute.TiledMma, left_t: cute.TiledMma,
                  lk: cute.ComposedLayout, la: cute.ComposedLayout, lat: cute.ComposedLayout,
                  H: cutlass.Constexpr, NC: cutlass.Constexpr, SCALE: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bh, n, b = bid % H, (bid // H) % NC, bid // (H * NC)
    smem = utils.SmemAllocator()
    def alloc(layout):
        return smem.allocate_tensor(cutlass.BFloat16, layout.outer, 128, swizzle=layout.inner)
    def transpose(t, layout):
        return cute.make_tensor(cute.recast_ptr(t.iterator, layout.inner, dtype=cutlass.BFloat16), layout.outer)
    sD0, sD1 = alloc(lk), alloc(lk)
    sA, sMp, sM, sX = alloc(la), alloc(la), alloc(la), alloc(la)
    sF = smem.allocate_tensor(cutlass.Float32, cute.make_layout((C, C), stride=(C + 1, 1)), byte_alignment=16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout(C), byte_alignment=16)
    sDb = smem.allocate_tensor(cutlass.Float32, cute.make_layout(C), byte_alignment=16)
    sAt, sXt = transpose(sA, lat), transpose(sX, lat)
    def load128(src, dst):
        for j in cutlass.range_constexpr(C * D // (NT * 8)):
            p = j * NT + tid
            r,c = p // (D//8),(p % (D//8))*8
            view = cute.make_tensor((src.iterator+src.layout((b,n,r,bh,c))).align(16),cute.make_layout(8))
            vec = cute.make_fragment_like(view)
            cute.autovec_copy(view,vec)
            off=(c//64)*4096+r*64+8*((c//8)%8 ^ (r%8))+c%8
            raw=cute.recast_ptr(dst.iterator,dtype=cutlass.BFloat16)
            packed=cute.make_tensor((raw+off).align(16),cute.make_layout(8))
            packed.store(vec.load())
    load128(do, sD0)
    load128(vd, sD1)
    for j in cutlass.range_constexpr(C * C // NT):
        p = j * NT + tid
        r, c = p // C, p % C
        sA[r, c, 0] = a[b, n, r, bh, c]
        sMp[r, c, 0] = mp[b, n, r, bh, c]
        sM[r, c, 0] = m[b, n, r, bh, c]
    if tid < C:
        sBeta[tid] = beta[b, n, tid, bh]
    cute.arch.fence_proxy('async.shared', space='cta')
    cute.arch.barrier()
    tp = gram.get_slice(tid)
    cp = tp.partition_C(cute.make_identity_tensor((C, C)))
    def gemm(mma, acc, sa, sb, accumulate=False):
        th = mma.get_slice(tid)
        fa = mma.make_fragment_A(th.partition_A(sa[None, None, 0]))
        fb = mma.make_fragment_B(th.partition_B(sb[None, None, 0]))
        warpgroup.fence()
        for j in cutlass.range(cute.size(fa, mode=[2]), unroll_full=True):
            mma.set(warpgroup.Field.ACCUMULATE, cutlass.Boolean(accumulate or j != 0))
            cute.gemm(mma, acc, fa[None, None, j], fb[None, None, j], acc)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    p = gram.make_fragment_C(gram.partition_shape_C((C, C)))
    dp = gram.make_fragment_C(gram.partition_shape_C((C, C)))
    gemm(gram, dp, sD0, sD1)                    # dO Vd^T
    pairf = cute.make_rmem_tensor((2,), cutlass.Float32)
    for jp in cutlass.range_constexpr(C * C // NT // 2):
        i = 2 * jp
        r, c = cp[i][0], cp[i][1]
        mv = cute.make_tensor((sMp.iterator + sMp.layout((r, c, 0))).align(4), cute.make_layout((2,))).load()
        m0 = cutlass.Float32(mv[0])
        m1 = cutlass.Float32(mv[1])
        dmpr0 = SCALE * pq[b, n, r, bh, c] * dp[i]
        dpr0 = SCALE * m0 * dp[i]
        dmpr1 = SCALE * pq[b, n, r, bh, c + 1] * dp[i + 1]
        dpr1 = SCALE * m1 * dp[i + 1]
        if c > r:
            dmpr0 = cutlass.Float32(0.)
            dpr0 = cutlass.Float32(0.)
        if c + 1 > r:
            dmpr1 = cutlass.Float32(0.)
            dpr1 = cutlass.Float32(0.)
        pairf[0], pairf[1] = dmpr0, dmpr1
        cute.make_tensor((dmp.iterator + dmp.layout((b, n, bh, r, c))).align(8), cute.make_layout((2,))).store(pairf.load())
        pairf[0], pairf[1] = dpr0, dpr1
        cute.make_tensor((dpq.iterator + dpq.layout((b, n, r, bh, c))).align(8), cute.make_layout((2,))).store(pairf.load())
    # dAb = dVd W^T; dbeta_col = column_sum(dAb * A).
    cute.arch.barrier()
    load128(dvd, sD0)
    load128(wv, sD1)
    cute.arch.fence_proxy('async.shared', space='cta')
    cute.arch.barrier()
    gemm(gram, dp, sD0, sD1)
    pairh2 = cute.make_rmem_tensor((2,), cutlass.BFloat16)
    for jp in cutlass.range_constexpr(C * C // NT // 2):
        i = 2 * jp
        r, c = cp[i][0], cp[i][1]
        pairh2[0], pairh2[1] = cutlass.BFloat16(dp[i] * sBeta[c]), cutlass.BFloat16(dp[i + 1] * sBeta[c + 1])
        cute.make_tensor((sX.iterator + sX.layout((r, c, 0))).align(4), cute.make_layout((2,))).store(pairh2.load())
    for j in cutlass.range_constexpr(C//8):
        i = 4*j
        r0,r1,c = cp[i][0],cp[i+2][0],cp[i][1]
        d0 = dp[i]*cutlass.Float32(sA[r0,c,0])+dp[i+2]*cutlass.Float32(sA[r1,c,0])
        d1 = dp[i+1]*cutlass.Float32(sA[r0,c+1,0])+dp[i+3]*cutlass.Float32(sA[r1,c+1,0])
        for off in (4,8,16):
            d0 = d0+cute.arch.shuffle_sync_bfly(d0,off)
            d1 = d1+cute.arch.shuffle_sync_bfly(d1,off)
        if tid%32<4:
            sF[tid//32,c] = d0
            sF[tid//32,c+1] = d1
    cute.arch.fence_proxy('async.shared', space='cta')
    cute.arch.barrier()
    if tid < C:
        val = cutlass.Float32(0.)
        for j in cutlass.range_constexpr(4):
            val = val + sF[j, tid]
        sDb[tid] = val
    # dLm = -A^T dA A^T (strict lower). sMp is now scratch Y.
    gemm(left_t, p, sAt, sXt)
    pairh3 = cute.make_rmem_tensor((2,), cutlass.BFloat16)
    for jp in cutlass.range_constexpr(C * C // NT // 2):
        i = 2 * jp
        pairh3[0], pairh3[1] = cutlass.BFloat16(p[i]), cutlass.BFloat16(p[i + 1])
        cute.make_tensor((sMp.iterator + sMp.layout((cp[i][0], cp[i][1], 0))).align(4), cute.make_layout((2,))).store(pairh3.load())
    cute.arch.fence_proxy('async.shared', space='cta')
    cute.arch.barrier()
    gemm(gram, dp, sMp, sA)
    for i in cutlass.range_constexpr(C * C // NT):
        p[i] = pkk[b, n, cp[i][0], bh, cp[i][1]]
    pairf4 = cute.make_rmem_tensor((2,), cutlass.Float32)
    for jp in cutlass.range_constexpr(C * C // NT // 2):
        i = 2 * jp
        r, c = cp[i][0], cp[i][1]
        mv4 = cute.make_tensor((sM.iterator + sM.layout((r, c, 0))).align(4), cute.make_layout((2,))).load()
        sm0 = cutlass.Float32(mv4[0])
        sm1 = cutlass.Float32(mv4[1])
        dl0 = cutlass.Float32(0.)
        dl1 = cutlass.Float32(0.)
        if c < r:
            dl0 = -dp[i]
        if c + 1 < r:
            dl1 = -dp[i + 1]
        pairf4[0], pairf4[1] = dl0 * sBeta[r] * p[i], dl1 * sBeta[r] * p[i + 1]
        cute.make_tensor((dm.iterator + dm.layout((b, n, bh, r, c))).align(8), cute.make_layout((2,))).store(pairf4.load())
        pairf4[0], pairf4[1] = dl0 * sBeta[r] * sm0, dl1 * sBeta[r] * sm1
        cute.make_tensor((dpkk.iterator + dpkk.layout((b, n, r, bh, c))).align(8), cute.make_layout((2,))).store(pairf4.load())
        p[i] = dl0 * p[i] * sm0
        p[i + 1] = dl1 * p[i + 1] * sm1
    d0,d1 = cutlass.Float32(0.),cutlass.Float32(0.)
    for i in cutlass.range_constexpr(C*C//NT):
        if cutlass.const_expr((i//2)%2==0):
            d0 = d0+p[i]
        else:
            d1 = d1+p[i]
    d0 = d0+cute.arch.shuffle_sync_bfly(d0,1)
    d0 = d0+cute.arch.shuffle_sync_bfly(d0,2)
    d1 = d1+cute.arch.shuffle_sync_bfly(d1,1)
    d1 = d1+cute.arch.shuffle_sync_bfly(d1,2)
    cute.arch.barrier()
    if tid%4==0:
        db[b,n,cp[0][0],bh] = sDb[cp[0][0]]+d0
        db[b,n,cp[2][0],bh] = sDb[cp[2][0]]+d1


@cute.jit
def launch_matrix(pq: cute.Tensor, pkk: cute.Tensor, do: cute.Tensor,
                  a: cute.Tensor, mp: cute.Tensor, m: cute.Tensor, beta: cute.Tensor,
                  wv: cute.Tensor, vd: cute.Tensor, dvd: cute.Tensor,
                  dpq: cute.Tensor, dpkk: cute.Tensor, db: cute.Tensor,
                  dmp: cute.Tensor, dm: cute.Tensor,
                  B: cutlass.Constexpr, H: cutlass.Constexpr, NC: cutlass.Constexpr, SCALE: cutlass.Constexpr,
                  stream: cuda.CUstream):
    bf, ff = cutlass.BFloat16, cutlass.Float32
    def mma(n, am, bm):
        return sm90.make_trivial_tiled_mma(bf, bf, am, bm, ff, (1, 1, 1), (C, n))
    gram = mma(C, OMM.K, OMM.K)
    left_t = mma(C, OMM.MN, OMM.MN)
    lk = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, C, D), bf, 1)
    la = sm90.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (C, C, C), bf, 1)
    lat = sm90.make_smem_layout_a(LayoutEnum.COL_MAJOR, (C, C, C), bf, 1)
    matrix_kernel(pq, pkk, do, a, mp, m, beta, wv, vd, dvd, dpq, dpkk, db, dmp, dm,
                  gram, left_t, lk, la, lat, H, NC, SCALE).launch(
        grid=(B * NC * H, 1, 1), block=(NT, 1, 1), stream=stream)


_compiled = {}


def kda_bwd_parallel(do, A, Mp, M, pq, pkk, gc, k2, q2, beta, W, Vd, dvd,
                     dgv_p, dk2v_p, dgo_p, dq2o_p, scale=None):
    """(dPq, dPkk (B,T,H,64) fp32, dgc (slice cumsum gradient), dk2, dq2, dbeta).  No packed documents."""
    B, T, H, D_ = do.shape
    E, NC = k2.shape[-1], T // C
    NR = dgv_p.shape[-2]
    assert D_ == D and T % C == 0
    scale = D ** -0.5 if scale is None else float(scale)
    dpq = torch.empty((B, T, H, C), device=do.device, dtype=torch.float32)
    dpkk = torch.empty_like(dpq)
    dg, dk2, dq2 = torch.empty_like(gc), torch.empty_like(k2), torch.empty_like(q2)
    db = torch.empty_like(beta)
    dmp = torch.empty((B, NC, H, C, C), device=do.device, dtype=torch.float32)
    dm = torch.empty_like(dmp)
    def chunks(t):
        return t.view(B, NC, C, *t.shape[2:])
    ma = [from_dlpack(chunks(t).detach(), assumed_align=16)
          for t in (pq, pkk, do, A, Mp, M, beta, W, Vd, dvd, dpq, dpkk, db)]
    masks = [from_dlpack(t.detach(), assumed_align=16) for t in (dmp, dm)]
    ra = masks + [from_dlpack(chunks(t).detach(), assumed_align=16)
                  for t in (gc, k2, q2, dgv_p, dk2v_p, dgo_p, dq2o_p, dg, dk2, dq2)]
    ra += [ma[0], ma[0]]                 # offsets, mapping: unused
    stream = cuda.CUstream(torch.cuda.current_stream(do.device).cuda_stream)
    key = (B, T, H, E, scale, NR, NC)
    if key not in _compiled:
        mat = cute.compile(launch_matrix, *ma, *masks, B, H, NC, scale, stream)
        mask_grad = cute.compile(launch_mask_grad, *ra, B, H, E, NC, NR, False, False, stream)
        _compiled[key] = (mat, mask_grad)
    mat, mask_grad = _compiled[key]
    mat(*ma, *masks, stream)
    mask_grad(*ra, stream)
    return dpq, dpkk, dg, dk2, dq2, db
