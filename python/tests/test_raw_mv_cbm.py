"""PLAN §4.2 mv:纯 raw eDSL mv,B 和 A 的 pack 都在 DSL 内完成(vpack(memref)→linalg.pack)。

mv: C[M] = B[M,K] @ A[K]。表达成 GEMM C = B @ Apadᵀ,Apad[Npad,K] 每行都是 A(n 维广播)。
B 侧、A 侧都用 vpack(memref) 在 kernel 内摆 cube 布局(→linalg.pack + collapse,见
probe_pack_cbm),替代原来的 host numpy pack_B / make_Acube。A 与 B 完全对称(同 mm)。

host 只做一件事:把 A[K] 广播成 Apad[Npad,K](np.broadcast_to,一行,非手摆 cube 布局)。
mv 真结果在 C 每行任意列(Apad 各行相同,取 col0)。

M=16,K=64,Npad=32,cube=8。b1=M/8=2,b2=Npad/8=4。对拍 torch.mv。
"""
import numpy as np, torch, triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

M, K, Npad = 16, 64, 32
KB = 8
KC = K // KB  # 8
B1, B2 = M // 8, Npad // 8  # 2, 4


@tle.raw_kernel
def mv(B: tle.mem(f16), Apad: tle.mem(f16), C: tle.mem(f16, out=True)):
    nvl = tle.vconfig(-1, 1)  # VL=64
    # B/A 侧都用 vpack(memref) 在 DSL 内摆 cube(行主序 → linalg.pack → collapse)
    Bcube = tle.vpack(B, inner_tiles=(16, 8), stride=K, rows=M)  # tensor<1×8×128>
    Acube = tle.vpack(Apad, inner_tiles=(32, 8), stride=K, rows=Npad)  # tensor<1×8×256>
    acc = tle.vzero(f32, group=8)  # <8×64xf32>
    for kc in tle.range(0, KC, 1):
        vb = tle.vload(Bcube, (0, kc), group=B1)  # <2×64xf16>
        va = tle.vload(Acube, (0, kc), group=B2)  # <4×64xf16>(A 已在 Apad 摊成 Npad 行)
        acc = tle.vmadot(acc, vb, va)  # cross_batch_matmul
    # 输出还原:vpack(vector) 逐级 group_interleave → 行主序
    c1 = tle.vpack(acc, 8)  # <8×64> → <4×128>
    c2 = tle.vpack(c1, 16)  # <4×128> → <2×256>
    cf = tle.vshape(c2, (16, 32))  # 行主序 <16×32xf32>
    c = tle.cast(cf, f16)
    tle.vstore(C, 0, c, shape=(16, 32))


@triton.jit
def host(B, Apad, C):
    _sr_call(mv, outputs=[], inputs=[B, Apad, C])


def test_mv_cbm():
    rng = np.random.default_rng(0)
    Blog = rng.standard_normal((M, K)).astype(np.float16)
    Alog = rng.standard_normal((K,)).astype(np.float16)
    golden = Blog.astype(np.float64) @ Alog.astype(np.float64)  # [M]
    B = torch.tensor(Blog.reshape(-1))  # 行主序,DSL 内 vpack 摆 cube
    # host 唯一预处理:A 广播成 Apad[Npad,K](每行都是 A),cube 布局交给 DSL vpack
    Apad = torch.tensor(np.broadcast_to(Alog, (Npad, K)).copy().reshape(-1))
    C = torch.zeros(M, Npad, dtype=torch.float16)
    host[(1,)](B.contiguous(), Apad.contiguous(), C)
    got = C[:, 0].float().numpy().astype(np.float64)  # mv 结果在每行 col0
    diff = np.abs(got - golden).max()
    assert diff < 5e-2, f"mv max_diff={diff:.4e}\ngot={got[:4]}\ngold={golden[:4]}"
