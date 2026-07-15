"""spine_raw svector-level mv (feishu 3.3 示例).

C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

style2 (纯 svector): vconfig/vzero/vload/vmacc/vreduce_sum/vstore, no packing.
style3 (svector + pack): the same, but B's 4-row block is pre-packed into a
        contiguous scratch buffer via tle.alloc + tle.pack before the K loop.

(矩阵单元 mv/mm 走 tle.vmadot → vector_ext.cross_batch_matmul,见
 test_raw_mm_cbm.py / test_raw_mv_cbm.py。)

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
            nvl = tle.vconfig(K - ki, 1)  # avl=K-ki:请求尾块收窄。avl 真收窄未落地时,
            #   codegen 降级为 valid=min(VLMAX, K-ki) + fill-0 pad(尾 lane 补 0),数值等价。
            va = tle.vload(A, ki)          # ← vload 自动按 vconfig 的 avl 补 0,无需写 valid=
            vb0 = tle.vload(B, ni * K + ki)
            vb1 = tle.vload(B, (ni + 1) * K + ki)
            vb2 = tle.vload(B, (ni + 2) * K + ki)
            vb3 = tle.vload(B, (ni + 3) * K + ki)
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


# ---------------------------------------------------------------------------
# 任意 K — kernel 内 padding 方案(零 host copy,style2)
# ---------------------------------------------------------------------------
# 约束一(K%64)在 kernel 内解决:K 尾块 vload valid=imin(64, K-ki) 走 fill-0 scratch
# (probe_svpad_fill 坐实),尾 lane 补 0,vmacc/vreduce_sum 补 0 无害。B/A 不做任何 host
# copy,kernel 直接吃真实 K。此处保持 N%4==0(约束二 N 尾另做,见 _SHAPES_ARB 说明)。
def _run_arb(host, N, K, BLOCK=4):
    assert N % 4 == 0, "本轮 kernel 内 padding 覆盖任意 K;N 尾(N%4)另做"
    B = torch.randn(N, K, dtype=torch.float16)     # 真实 K, 不 pad
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(N, dtype=torch.float32)
    grid = (N // BLOCK, )
    host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    ref = torch.mv(B.float(), A.float())
    max_diff = (C - ref).abs().max().item()
    assert torch.allclose(C, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"


# 任意 K(N%4==0):K 非 64 倍数, kernel 内 fill-0 补 K 尾
_SHAPES_ARB = [(4, 60), (4, 100), (8, 65), (4, 63), (8, 127), (4, 200), (16, 130), (12, 50)]


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style2_arb(N, K):
    _run_arb(_mv_sv_host_style2, N, K)

