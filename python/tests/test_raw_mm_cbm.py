"""PLAN §4.2:纯 raw eDSL(非注入 _mlir_text)的最小 mm,走 vmadot→cross_batch_matmul。

C[M,N] = A[M,K] @ B[N,K]ᵀ,单 (mc=1,nc=1) block:M=16,N=32,K=64,cube=8。
host 端按 linalg.pack 规则把 A/B 摆成 packed 连续 buffer(与 probe_cbm_e2e 同,cbm 输入
吃平铺 pack,不需输入侧 vpack);kernel 用 spine_raw 原语:
  逐 kc-tile: vload(group=) 读 packed 连续 → vmadot 累加(cross_batch_matmul)
  末: vpack×2 (group_interleave 还原) → vshape → vstore
对拍 torch A@Bᵀ。这是把手写 MLIR probe 升级成 codegen 真生成的关键验证。
"""
import numpy as np, torch, triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

M, N, K = 16, 32, 64
MB, NB, KB = 16, 32, 8
KC = K // KB  # 8
B1, B2 = MB // 8, NB // 8  # 2, 4


@tle.raw_kernel
def mm(Ap: tle.mem(f16), Bp: tle.mem(f16), C: tle.mem(f16, out=True)):
    # Ap packed <1,8,16,8> flat, Bp packed <1,8,32,8> flat. per kc-tile:
    #   A tile 连续 128 = <2×64>(b1=2 cubes), B tile 连续 256 = <4×64>(b2=4)
    nvl = tle.vconfig(-1, 1)  # VL=64
    acc = tle.vzero(f32, group=8)  # <8×64xf32> = b1·b2
    for kc in tle.range(0, KC, 1):
        va = tle.vload(Ap, kc * 128, group=B1)  # <2×64xf16>
        vb = tle.vload(Bp, kc * 256, group=B2)  # <4×64xf16>
        acc = tle.vmadot(acc, va, vb)  # cross_batch_matmul
    c1 = tle.vpack(acc, 8)  # <8×64> → <4×128>
    c2 = tle.vpack(c1, 16)  # <4×128> → <2×256>
    cf = tle.vshape(c2, (16, 32))  # → 行主序 <16×32xf32>
    c = tle.cast(cf, f16)  # f32 → f16(匹配 C)
    tle.vstore(C, 0, c, shape=(16, 32))  # 2D 块写回(1D 宽向量 transfer_write 会丢 lane)


@triton.jit
def host(Ap, Bp, C):
    _sr_call(mm, outputs=[], inputs=[Ap, Bp, C])


def pack_A(Alog):  # [1,kc,16,8]
    P = np.zeros((1, KC, MB, KB), np.float16)
    for kc in range(KC):
        for mb in range(MB):
            for kb in range(KB):
                P[0, kc, mb, kb] = Alog[mb, kc * KB + kb]
    return P


def pack_B(Blog):  # [1,kc,32,8]
    P = np.zeros((1, KC, NB, KB), np.float16)
    for kc in range(KC):
        for nb in range(NB):
            for kb in range(KB):
                P[0, kc, nb, kb] = Blog[nb, kc * KB + kb]
    return P


def test_mm_cbm():
    rng = np.random.default_rng(0)
    Alog = rng.standard_normal((M, K)).astype(np.float16)
    Blog = rng.standard_normal((N, K)).astype(np.float16)
    golden = Alog.astype(np.float64) @ Blog.astype(np.float64).T
    Ap = torch.tensor(pack_A(Alog).reshape(-1))
    Bp = torch.tensor(pack_B(Blog).reshape(-1))
    C = torch.zeros(M, N, dtype=torch.float16)
    host[(1,)](Ap.contiguous(), Bp.contiguous(), C)
    out = C.float().numpy().astype(np.float64)
    diff = np.abs(out - golden).max()
    assert diff < 5e-2, f"mm max_diff={diff:.4e}\nout[0,:4]={out[0,:4]}\ngold={golden[0,:4]}"
