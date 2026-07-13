"""PLAN §4.2 mv:纯 raw eDSL mv,B 和 A 的 pack 都在 DSL 内、零多搬。

mv: C[M] = B[M,K] @ A[K]。表达成 GEMM,A 是被广播维(cube 的 n 维退化)。
- B 侧:vpack(memref)→linalg.pack 摆 cube(见 probe_pack_cbm)。
- A 侧(两层广播,PLAN §3.6):
  ① k→cube 的 n 广播:tle.spread(A, (KC,8,8)) —— scf.for 标量 memref.store,只读 A 的
     K 个原始数,写进 <KC,8,8> scratch(不产 vector→绕开 vscale)。A 只传原始 <K> 向量,零多搬。
  ② cube→b2 份:tle.vbroadcast(va1, B2) 寄存器复制(vector.broadcast <64>→<4×64>)。

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
def mv(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f16, out=True)):
    nvl = tle.vconfig(-1, 1)  # VL=64
    # ① B 侧:vpack(memref) 在 DSL 内摆 cube 布局(行主序 → linalg.pack → collapse)
    Bcube = tle.vpack(B, inner_tiles=(16, 8), stride=K, rows=M)  # tensor<1×8×128>
    # ② A 侧:spread 软件广播成 cube scratch(scf.for 标量, 无 vector→无 vscale)
    scrA = tle.spread(A, cube_shape=(KC, 8, 8))  # memref<8×64>(每 kc 一个已 n 广播的 cube)
    acc = tle.vzero(f32, group=8)  # <8×64xf32>
    for kc in tle.range(0, KC, 1):
        vb = tle.vload(Bcube, (0, kc), group=B1)  # <2×64xf16>
        va1 = tle.vload(scrA, (kc, 0))  # <64xf16>(单 cube,A 已 n 广播)
        va = tle.vbroadcast(va1, B2)  # <4×64xf16> 寄存器复制 b2 份
        acc = tle.vmadot(acc, vb, va)  # cross_batch_matmul
    # ⑤ 输出还原:vpack(vector) 逐级 group_interleave → 行主序
    c1 = tle.vpack(acc, 8)  # <8×64> → <4×128>
    c2 = tle.vpack(c1, 16)  # <4×128> → <2×256>
    cf = tle.vshape(c2, (16, 32))  # 行主序 <16×32xf32>
    c = tle.cast(cf, f16)
    tle.vstore(C, 0, c, shape=(16, 32))


@triton.jit
def host(B, A, C):
    _sr_call(mv, outputs=[], inputs=[B, A, C])


def test_mv_cbm():
    rng = np.random.default_rng(0)
    Blog = rng.standard_normal((M, K)).astype(np.float16)
    Alog = rng.standard_normal((K,)).astype(np.float16)
    golden = Blog.astype(np.float64) @ Alog.astype(np.float64)  # [M]
    B = torch.tensor(Blog.reshape(-1))  # 行主序,DSL 内 vpack 摆 cube
    A = torch.tensor(Alog.reshape(-1))  # 原始 <K> 向量,DSL 内 spread 广播,零 host 预处理
    C = torch.zeros(M, Npad, dtype=torch.float16)
    host[(1,)](B.contiguous(), A.contiguous(), C)
    got = C[:, 0].float().numpy().astype(np.float64)  # mv 结果在每行 col0
    diff = np.abs(got - golden).max()
    assert diff < 5e-2, f"mv max_diff={diff:.4e}\ngot={got[:4]}\ngold={golden[:4]}"