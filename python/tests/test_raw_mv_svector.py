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
# 写法4 — 文档 3.3 第三段的矩阵单元写法 (tle.vmadot)
#
# 结构对齐文档: 签名 (B, A, C, K, N)、循环外 tle.alloc 建 packed 缓冲、
# 内层 `for ni in range(0,N,·)` + `for ki in range(0,K,·)`、tle.vpack 打包 B、
# tle.vload 取块、tle.vmadot(acc, vb, va) 矩阵单元累加(acc 在前, 直接产宽
# 结果, 不需 vreduce_sum)、tle.vstore 写回。
#
# spine-triton 编译器把 tle.vmadot lower 到已支持的 vector_ext.batch_macc
# (vfwmacc, 走 spe_pack)——不是未定稿的 vector_ext.matmul,这正是「编译器
# 支持时可以用 batch_macc」的含义(_gen_vmadot 见 codegen.py)。
#
# 与文档字面的必要差异(来自 batch_macc 契约, 不可约): batch_macc 要求
# n%64==0, 故 output 行块 NB=64(文档写 8)、单 acc(文档 4 个 8 宽 acc);
# 且 lhs 必须是 2D strided memref, 故 A(向量)作 [1,32] memref 视图、B 转置
# pack 成 [32,64] 向量作 rhs。约束 N%64==0 且 K%32==0。
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style4(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, N: tle.index):
    packed_B = tle.alloc_tcm_2d(32, 64, "f16")
    for ni in tle.range(0, N, 64):
        acc0 = tle.splat_2d(0.0, 1, 64, "f32")
        for ki in tle.range(0, K, 32):
            va = tle.view_2d(A, 1, 32, "f16", ki)
            tle.pack_2d_t_into(packed_B, B, ni, 32, 64, K, "f16", ki)
            vb0 = tle.load_2d(packed_B, 32, 64, "f16")
            acc0 = tle.vmadot(acc0, va, vb0)
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
