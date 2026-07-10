"""spine_raw svector-level mv (feishu 3.3 示例).

C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

style2 (纯 svector): vconfig/vzero/vload/vmacc/vreduce_sum/vstore, no packing.
style3 (svector + pack): the same, but B's 4-row block is pre-packed into a
        contiguous scratch buffer via tle.alloc + tle.pack before the K loop.
style4 (矩阵单元 mmt4d): mv 表达成 GEMM，走结构化 linalg.pack+mmt4d+unpack，
        下游 spe_pack 自动生成 cube 布局 → smt.vfwmadot（tle.mmt4d）。数值正确，
        Nrow%16==0、K%8==0。

Fixed VL (f16 -> 64) this round: no dynamic vsetvl tail handling, so the tests
constrain K % 64 == 0 and N % 4 == 0 (full tiles only).

访存按 SPEC §6.2 规格:vload(ptr, index)/vstore(ptr, index, value),index 为扁平
标量元素偏移,二维坐标由用户自行压平(如 B 的行 ni 列 ki 写作 ni*K + ki)。对
alloc 出的 ranked scratch(packed_B),index 仍是逐维下标元组(ranked 自然寻址)。
"""
import functools

import torch
import triton
import triton.language as tl
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
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, row_base: tle.index,
                    row_end: tle.index):
    # grid 并发:host 按 program_id 把 N 行切块,本 program 只算 [row_base, row_end) 行。
    nvl = tle.vconfig(-1, 1)   # lmul=1 → VLMAX=64 (f16, SPEC §6.1)
    for ni in tle.range(row_base, row_end, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        for ki in tle.range(0, K, nvl):
            nvl = tle.vconfig(K - ki, 1)  # avl=K-ki (tail narrowing deferred), lmul=1
            vb0 = tle.vload(B, ni * K + ki)
            vb1 = tle.vload(B, (ni + 1) * K + ki)
            vb2 = tle.vload(B, (ni + 2) * K + ki)
            vb3 = tle.vload(B, (ni + 3) * K + ki)
            va = tle.vload(A, ki)
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        tle.vstore(C, ni, tle.vreduce_sum(acc0))
        tle.vstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.vstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.vstore(C, ni + 3, tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style2(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = row_base + BLOCK
    _sr_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, row_base, row_end])


# ---------------------------------------------------------------------------
# 写法3 — svector + tle.alloc/tle.pack 预打包 B
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style3(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, row_base: tle.index,
                    row_end: tle.index):
    # grid 并发:本 program 只算 [row_base, row_end) 行。
    nvl = tle.vconfig(-1, 1)   # lmul=1 → VLMAX=64 (f16, SPEC §6.1)
    packed_B = tle.alloc((1, K // nvl, 4, nvl), f16)
    for ni in tle.range(row_base, row_end, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        tle.pack(B, (ni, 0), packed_B, (1, K // nvl, 4, nvl), K)
        for ki in tle.range(0, K, nvl):
            nvl = tle.vconfig(K - ki, 1)  # avl=K-ki (tail narrowing deferred), lmul=1
            vb0 = tle.vload(packed_B, (0, ki // nvl, 0, 0))
            vb1 = tle.vload(packed_B, (0, ki // nvl, 1, 0))
            vb2 = tle.vload(packed_B, (0, ki // nvl, 2, 0))
            vb3 = tle.vload(packed_B, (0, ki // nvl, 3, 0))
            va = tle.vload(A, ki)
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        tle.vstore(C, ni, tle.vreduce_sum(acc0))
        tle.vstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.vstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.vstore(C, ni + 3, tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style3(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = row_base + BLOCK
    _sr_call(mv_block_style3, outputs=[], inputs=[B, A, C, K, row_base, row_end])


# ---------------------------------------------------------------------------
# 写法4 — svector + tle.vfwmadot 矩阵单元(feishu 3.3 第三段, vfwmadot 风格)
#
# 矩阵引擎指令 vfwmadot 直接产出宽结果, 不需 vreduce_sum。映射到 K3 已注册的
# vector_ext.matmul (m=n=k=8 tile) → llvm.riscv.smt.vfwmadot (xsmtvdotii mattr)。
# operand 为整寄存器宽 (f16=64, f32=64), 与仓库 mma_gen.mlir 实验一致。
# ---------------------------------------------------------------------------
#
# 数值正确的实现:mv = GEMM，走结构化 linalg.pack+mmt4d+unpack，下游 spe_pack
# 自动生成 cube 布局 → smt.vfwmadot（不用前端手推 lane）。C[Nrow,K]·A[K] 表达成
# C2[Nrow,32] = B[Nrow,K] @ Apad[K,32]（A 放第 0 列），mv 结果取 C2[:,0]。
# tile mb=16/nb=32/kb=8：Nrow%16==0，K%8==0。已在 K3(179) 对拍 torch.mv max_diff~7e-3。
@functools.lru_cache(maxsize=None)
def _make_style4_host(N, K):
    # 文档 §3.3 第三段的 svector 写法4 surface:packed_B=tle.alloc + tle.vpack 打包 B,
    # 内层 for ki 逐 8-K-tile 取 vb/va 用 tle.vfwmadot 矩阵单元累加。codegen 把这套
    # 「vfwmadot 循环」pattern 整体折成结构化 linalg.pack+mmt4d+unpack(cube 布局交下游
    # spe_pack 自动生成),底层复用已在 K3 对拍 torch.mv 通过的 mmt4d 路。N/K 经闭包烘成
    # 编译期常量(mmt4d 需固定维度)。每 shape 唯一 __name__ 避免 Triton JIT 按名缓存串用。
    # grid 并发:每 program 算 BLOCK 行,codegen 把 vfwmadot 循环折成 mmt4d,M=BLOCK(编译期
    # 闭包常量)、B/C 按 row_base(=pid*BLOCK)动态偏移。row_base 形参驱动 fold 的动态 offset。
    @tle.raw_kernel
    def mv_block_style4(B: tle.mem(f16), Apad: tle.mem(f16), C2: tle.mem(f16, out=True), row_base: tle.index):
        packed_B = tle.alloc((1, K // 8, 32, 8), f16)
        for ni in tle.range(row_base, row_base + N, 32):
            acc = tle.vzero(f32)
            tle.vpack(B, (ni, 0), packed_B, (1, K // 8, 32, 8))
            for ki in tle.range(0, K, 8):
                vb = tle.vload(packed_B, (0, ki // 8, 0, 0))
                va = tle.vload(Apad, ki)
                acc = tle.vfwmadot(acc, vb, va)
            tle.vstore(C2, ni, acc)

    mv_block_style4._fn.__name__ = f"mv_block_style4_{N}_{K}"

    @triton.jit
    def host(B, Apad, C2, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        _sr_call(mv_block_style4, outputs=[], inputs=[B, Apad, C2, row_base])

    host.__name__ = f"_mv_host_style4_{N}_{K}"
    host.fn.__name__ = host.__name__
    return host


def _run_style4(N, K, BLOCK=16):
    # mv: C[N] = B[N,K] @ A[K]。pad 成 GEMM，A 放第 0 列，取输出第 0 列。
    # grid 并发:N 行按 BLOCK(须%16,mmt4d mb tile)切成 N//BLOCK 个 program。
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    Apad = torch.zeros(K, 32, dtype=torch.float16)
    Apad[:, 0] = A
    C2 = torch.zeros(N, 32, dtype=torch.float16)
    grid = (N // BLOCK, )
    _make_style4_host(BLOCK, K)[grid](B.contiguous(), Apad.contiguous(), C2, BLOCK=BLOCK)
    got = C2[:, 0].float()
    ref = torch.mv(B.float(), A.float())
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), f"N={N} K={K} max_diff={max_diff:.4e}"


def _run(host, N, K, BLOCK=4):
    # grid 并发:N 行按 BLOCK 切成 N//BLOCK 个 program(每个算 BLOCK 行,内层 4 行一组)。
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(N, dtype=torch.float32)
    grid = (N // BLOCK, )
    host[grid](B.contiguous(), A.contiguous(), C, K, N, BLOCK=BLOCK)
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


# 写法4 (mmt4d 矩阵单元): Nrow 须为 16 的倍数 (mb tile), K 须为 8 的倍数 (kb tile)。
_SHAPES_MADOT = [(16, 64), (32, 128), (64, 256), (128, 512)]


@pytest.mark.parametrize("N, K", _SHAPES_MADOT)
def test_raw_mv_svector_style4(N, K):
    _run_style4(N, K)
