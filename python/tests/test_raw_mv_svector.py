"""spine_raw svector-level mv (feishu 3.3 示例).

C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

style2 (纯 svector): vconfig/vzero/vload/vmacc/vreduce_sum/vstore, no packing.
style3 (svector + pack): the same, but B's 4-row block is pre-packed into a
        contiguous scratch buffer via tle.alloc + tle.vpack before the K loop.
style4 (矩阵单元 tle.vmadot): kernel 用文档 3.3 第三段的 tle.vmadot 矩阵单元
        算子 surface(acc 在前, 直接产出 1×64 宽结果, 不需 vreduce_sum)。
        spine-triton 编译器把 vmadot lower 到已支持的 vector_ext.batch_macc
        (vfwmacc, 走 spe_pack)——而非未定稿的 vector_ext.matmul。N%64==0 且
        K%32==0。

Fixed VL (f16 -> 64) this round: no dynamic vsetvl tail handling, so the tests
constrain K % 64 == 0 and N % 4 == 0 (full tiles only).

Note (minor deviation from the doc surface): tle.vload of a 2D index into an
external pointer needs the matrix row stride, so B loads pass it explicitly as
`tle.vload(B, (ni, ki), K)`. Everything else matches the document.
"""
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32


# ---------------------------------------------------------------------------
# 写法2 — 纯 svector
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, N: tle.index):
    nvl = tle.vconfig(-1, 2)
    for ni in tle.range(0, N, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        for ki in tle.range(0, K, nvl):
            nvl = tle.vconfig(K - ki, 2)
            vb0 = tle.vload(B, (ni, ki), K)
            vb1 = tle.vload(B, (ni + 1, ki), K)
            vb2 = tle.vload(B, (ni + 2, ki), K)
            vb3 = tle.vload(B, (ni + 3, ki), K)
            va = tle.vload(A, (ki, ))
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        tle.vstore(C, (ni, ), tle.vreduce_sum(acc0))
        tle.vstore(C, (ni + 1, ), tle.vreduce_sum(acc1))
        tle.vstore(C, (ni + 2, ), tle.vreduce_sum(acc2))
        tle.vstore(C, (ni + 3, ), tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style2(B, A, C, K, N):
    _sr_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, N])


# ---------------------------------------------------------------------------
# 写法3 — svector + tle.alloc/tle.vpack 预打包 B
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style3(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, N: tle.index):
    nvl = tle.vconfig(-1, 2)
    packed_B = tle.alloc((1, K // nvl, 4, nvl), f16)
    for ni in tle.range(0, N, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        tle.vpack(B, (ni, 0), packed_B, (1, K // nvl, 4, nvl), K)
        for ki in tle.range(0, K, nvl):
            nvl = tle.vconfig(K - ki, 2)
            vb0 = tle.vload(packed_B, (0, ki // nvl, 0, 0))
            vb1 = tle.vload(packed_B, (0, ki // nvl, 1, 0))
            vb2 = tle.vload(packed_B, (0, ki // nvl, 2, 0))
            vb3 = tle.vload(packed_B, (0, ki // nvl, 3, 0))
            va = tle.vload(A, (ki, ))
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        tle.vstore(C, (ni, ), tle.vreduce_sum(acc0))
        tle.vstore(C, (ni + 1, ), tle.vreduce_sum(acc1))
        tle.vstore(C, (ni + 2, ), tle.vreduce_sum(acc2))
        tle.vstore(C, (ni + 3, ), tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style3(B, A, C, K, N):
    _sr_call(mv_block_style3, outputs=[], inputs=[B, A, C, K, N])


# ---------------------------------------------------------------------------
# 写法4 — 文档 3.3 第三段的矩阵单元写法 (tle.vmadot),寄存器级广播零搬运
#
# 结构对齐文档: 签名 (B, A, C, K, N)、内层 for ni + for ki、tle.vmadot(acc, va, vb)
# 矩阵单元累加(直接产宽结果, 不需 vreduce_sum)。
#
# 【只多算不多搬运 + 寄存器级广播】关键取舍:
#   - lhs = va = view_2d(A, 1, 32): A(向量)的 memref 视图, 零拷贝。它的标量
#     在 vfwmacc 里由矩阵单元【寄存器级广播】乘 rhs——这是「多算」的那侧。
#   - rhs = vb = load_2d_t(B, ...): B(矩阵)的【转置 strided 读】(strides=[1,K]
#     → RVV vlse gather), 直接从 B 内存读出 [32,64], 不经 pack/TCM staging
#     buffer——【不多搬运】。对比旧写法用 pack_2d_t_into 把 B 物理转置进 TCM
#     buffer(多搬运), 本写法去掉了它。
#   - vmadot → vector_ext.batch_macc(vfwmacc), acc[1,n]+=va[1,k]·vb[k,n]。
#
# batch_macc 契约: rhs/acc 的 n 维需 %numelPerVReg(K3 f16=64)==0, 故 NB=64;
# lhs 须 2D strided memref(view_2d 满足)。约束 N%64==0 且 K%32==0。
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style4(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, N: tle.index):
    for ni in tle.range(0, N, 64):
        acc0 = tle.splat_2d(0.0, 1, 64, "f32")
        for ki in tle.range(0, K, 32):
            va = tle.view_2d(A, 1, 32, "f16", ki)  # A 向量视图 → 寄存器广播 lhs
            vb0 = tle.load_2d_t(B, ni, 32, 64, K, "f16", ki)  # B 转置 strided 读(vlse, 零搬运)
            acc0 = tle.vmadot(acc0, va, vb0)  # 矩阵单元 → vector_ext.batch_macc
        tle.store_2d_at(C, ni, 1, 64, acc0)


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style4(B, A, C, K, N):
    _sr_call(mv_block_style4, outputs=[], inputs=[B, A, C, K, N])


def _run(host, N, K):
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(N, dtype=torch.float32)
    host[(1, )](B.contiguous(), A.contiguous(), C, K, N)
    ref = torch.mv(B.float(), A.float())
    max_diff = (C - ref).abs().max().item()
    assert torch.allclose(C, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"


_SHAPES = [(4, 64), (8, 128), (16, 256), (32, 512), (64, 64), (128, 256)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style2(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style3(N, K):
    _run(_mv_sv_host_style3, N, K)


def _run_style4(N, K):
    # 写法4 用 (1,) grid + 与 style2/3 同签名, N-loop 在 kernel 内。
    B = torch.randn(N, K, dtype=torch.float16)  # matrix
    A = torch.randn(K, dtype=torch.float16)  # vector
    C = torch.empty(N, dtype=torch.float32)
    _mv_sv_host_style4[(1, )](B.contiguous(), A.contiguous(), C, K, N)
    ref = torch.mv(B.float(), A.float())
    max_diff = (C - ref).abs().max().item()
    assert torch.allclose(C, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"


# 写法4 (tle.vmadot → vector_ext.batch_macc): N%64==0 且 K%32==0 (满 tile)。
_SHAPES_MADOT = [(64, 32), (64, 64), (128, 32), (256, 64), (128, 256)]


@pytest.mark.parametrize("N, K", _SHAPES_MADOT)
def test_raw_mv_svector_style4(N, K):
    _run_style4(N, K)
