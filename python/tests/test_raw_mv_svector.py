"""spine_raw svector-level mv (feishu 3.3 示例).

C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

style2 (纯 svector): vconfig/vzero/vload/vmacc/vreduce_sum/vstore, no packing.
style3 (svector + pack): the same, but B's 4-row block is pre-packed into a
        contiguous scratch buffer via tle.alloc + tle.pack before the K loop.
style4 (svector + vfwmadot 矩阵单元): matrix-engine dot (tle.vfwmadot →
        vector_ext.matmul, m=n=k=8 → llvm.riscv.smt.vfwmadot) produces the wide
        result directly, no vreduce_sum. N must be a multiple of 8.

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
# 写法3 — svector + tle.alloc/tle.pack 预打包 B
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
        tle.pack(B, (ni, 0), packed_B, (1, K // nvl, 4, nvl), K)
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
# 写法4 — svector + tle.vfwmadot 矩阵单元(feishu 3.3 第三段, vfwmadot 风格)
#
# 矩阵引擎指令 vfwmadot 直接产出宽结果, 不需 vreduce_sum。映射到 K3 已注册的
# vector_ext.matmul (m=n=k=8 tile) → llvm.riscv.smt.vfwmadot (xsmtvdotii mattr)。
# operand 为整寄存器宽 (f16=64, f32=64), 与仓库 mma_gen.mlir 实验一致。
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style4(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, N: tle.index):
    nvl = tle.vconfig(-1, 2)
    for ni in tle.range(0, N, 8):
        acc = tle.vzero(f32)
        for ki in tle.range(0, K, nvl):
            nvl = tle.vconfig(K - ki, 2)
            vb = tle.vload(B, (ni, ki), K)
            va = tle.vload(A, (ki, ))
            acc = tle.vfwmadot(acc, vb, va)
        tle.vstore(C, (ni, ), acc)


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


# 写法4 (vfwmadot 矩阵单元): N 须为 8 的倍数 (matmul tile m=8)。
_SHAPES_MADOT = [(8, 64), (16, 128), (32, 256), (64, 512)]


@pytest.mark.parametrize("N, K", _SHAPES_MADOT)
@pytest.mark.xfail(
    reason="写法4 前端 (tle.vfwmadot → vector_ext.matmul) 已实现并在 K3 "
    "干净 lower 到 llvm.riscv.smt.vfwmadot(无 legalize 失败);但 mv→GEMM "
    "的 8×8×8 tile 映射数值待对齐:vector_ext.matmul 是把 64 宽 operand "
    "当 8×8 行主 cube 的整块 GEMM(out[m,n]=Σ_k lhs[m,k]·rhs[n,k]),正确 "
    "mv 需 lhs=B 的 8×8 strided tile、rhs=A 广播、输出取第 0 列 strided "
    "extract。该 tile lane 语义 spine-mlir 尚未定稿(docs 标注『待与提案人对齐』,"
    "仓库 mma512.mlir 仍在探测 lane),故本轮先落前端、数值标 xfail。", strict=False)
def test_raw_mv_svector_style4(N, K):
    _run(_mv_sv_host_style4, N, K)
