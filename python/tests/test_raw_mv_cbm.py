"""PLAN §4.2 mv:纯 raw eDSL mv,走 vmadot→cross_batch_matmul + vbroadcast(广播维)。

mv: C[M] = B[M,K] @ A[K]。表达成 GEMM,A 是被广播维(cube 的 n 维退化)。
广播策略:寄存器内 vbroadcast(省内存 padding 多存多搬,只剩 cube 多算)。

M=16,K=64,cube=8。b1=MB/8=2,b2=Npad/8=4(Npad=32 避开 scalable 边界)。
host 预处理(numpy):
  - Bp: B[16,K] pack 成 <1,kc,16,8>(和 mm 同)
  - Acube: 每 kc-tile 的 A[8] 沿 n 广播成单 cube <8×8>=64(host numpy 做 A 的 n 广播)
kernel:
  - vb = vload(Bp, group=b1)  <2×64>
  - va1 = vload(Acube)  <64>(单 cube) → vbroadcast(b2)  <4×64>
  - vmadot 累加,输出 vpack×2 + vshape 还原,取 mv 结果(每行任意列, 取 col0)
对拍 torch.mv。

诚实边界(PLAN §3.6):A 的「k→单cube n 广播」是 host numpy 做的;kernel 内只做
「单cube→b2 份」的 vbroadcast(寄存器)。这与 probe_mv_regbcast 一致。
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
MB, KB = 16, 8
KC = K // KB  # 8
B1, B2 = MB // 8, Npad // 8  # 2, 4


@tle.raw_kernel
def mv(Bp: tle.mem(f16), Acube: tle.mem(f16), C: tle.mem(f16, out=True)):
    nvl = tle.vconfig(-1, 1)  # VL=64
    acc = tle.vzero(f32, group=8)  # <8×64xf32>
    for kc in tle.range(0, KC, 1):
        vb = tle.vload(Bp, kc * 128, group=B1)  # <2×64xf16>
        va1 = tle.vload(Acube, kc * 64)  # <64xf16> 单 cube(A 沿 n 已广播)
        va = tle.vbroadcast(va1, B2)  # <4×64xf16> 寄存器广播
        acc = tle.vmadot(acc, vb, va)  # cross_batch_matmul
    c1 = tle.vpack(acc, 8)  # <8×64> → <4×128>
    c2 = tle.vpack(c1, 16)  # <4×128> → <2×256>
    cf = tle.vshape(c2, (16, 32))  # 行主序 <16×32xf32>
    c = tle.cast(cf, f16)
    tle.vstore(C, 0, c, shape=(16, 32))


@triton.jit
def host(Bp, Acube, C):
    _sr_call(mv, outputs=[], inputs=[Bp, Acube, C])


def pack_B(Blog):  # B[M,K] -> <1,kc,16,8>
    P = np.zeros((1, KC, MB, KB), np.float16)
    for kc in range(KC):
        for mb in range(MB):
            for kb in range(KB):
                P[0, kc, mb, kb] = Blog[mb, kc * KB + kb]
    return P


def make_Acube(Alog):
    # 每 kc-tile 的 A[8] 沿 n 广播成 <8×8> cube(n=8 行都是同一个 A[8] k-tile),展平 64。
    # cube 布局须与 cbm 对 rhs 的 <b2×64> 消费一致:这里 b2=4 由 kernel vbroadcast 生成,
    # host 只造单 cube(1 份 <8×8>)。cube[n_lane, k_lane] = A[kc*8 + k_lane](n 广播)。
    P = np.zeros((KC, 8, 8), np.float16)
    for kc in range(KC):
        for nl in range(8):
            for kl in range(8):
                P[kc, nl, kl] = Alog[kc * KB + kl]
    return P  # (kc, 8, 8) → flat per kc = 64


def test_mv_cbm():
    rng = np.random.default_rng(0)
    Blog = rng.standard_normal((M, K)).astype(np.float16)
    Alog = rng.standard_normal((K,)).astype(np.float16)
    golden = Blog.astype(np.float64) @ Alog.astype(np.float64)  # [M]
    Bp = torch.tensor(pack_B(Blog).reshape(-1))
    Acube = torch.tensor(make_Acube(Alog).reshape(-1))
    C = torch.zeros(M, Npad, dtype=torch.float16)
    host[(1,)](Bp.contiguous(), Acube.contiguous(), C)
    got = C[:, 0].float().numpy().astype(np.float64)  # mv 结果在每行 col0
    diff = np.abs(got - golden).max()
    assert diff < 5e-2, f"mv max_diff={diff:.4e}\ngot={got[:4]}\ngold={golden[:4]}"
