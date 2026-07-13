"""PLAN §4.2 mv:纯 raw eDSL mv,B/A 的 pack 都在 DSL 内、零多搬,且 grid 多线程并发。

mv: C[M] = B[M,K] @ A[K]。表达成 GEMM,A 是被广播维(cube 的 n 维退化)。
- B 侧:vpack(memref)→linalg.pack 摆 cube(见 probe_pack_cbm)。
- A 侧(两层广播,PLAN §3.6):
  ① k→cube 的 n 广播:tle.spread(A,(KC,CN,CK)) —— scf.for 标量 memref.store,只读 A 的
     K 个原始数,写进 <KC,CN,CK> scratch(不产 vector→绕开 vscale)。A 只传原始 <K> 向量,零多搬。
  ② cube→b2 份:tle.vbroadcast(va1, B2) 寄存器复制(vector.broadcast <64>→<b2×64>)。

grid 并发:M 行按 MB(=一个 cube 行块)切成 M//MB 个 program,K3 num_threads=4 起多线程。
每 program 用 tl.program_id 得 row_base,只 pack/算/写自己的 [row_base, row_base+MB) 行块:
  - B 侧 vpack(offset=row_base*K):reinterpret_cast 从 row_base*K 起,只 pack 本块 <MB×K>。
  - C 侧 vstore(index=row_base*Npad):写回本块。
  - A 侧 spread 与行块无关(每 program 读同一个 A[K])。

cube 尺寸从 dtype 的 MMACubicSize 推(K3 f16={m8,n8,k8}),不硬编码 8。
"""
import numpy as np, torch, triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language as tl
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

# MMA cube dims from the target dtype (mirror of TargetDescriptionAnalysis
# getMMACubicSize; K3 f16 = {m8,n8,k8}), not hardcoded 8.
CM, CN, CK = tle.mma_cube(f16)  # (8, 8, 8) for f16

M, K, Npad = 64, 64, 32          # M=64 → 多个 MB 行块(grid 并发)
MB = 16                          # 每 program 一个 cube 行块(rt=MB, b1=MB/CM=2)
KB = CK
KC = K // KB                     # 8
B1, B2 = MB // CM, Npad // CN     # 2, 4


@tle.raw_kernel
def mv(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f16, out=True), row_base: tle.index):
    nvl = tle.vconfig(-1, 1)  # VL=64
    # ① B 侧:vpack(memref, offset=row_base*K) 只摆本 program 的 MB 行块 cube
    Bcube = tle.vpack(B, inner_tiles=(MB, CK), stride=K, rows=MB, offset=row_base * K)
    # ② A 侧:spread 软件广播成 cube scratch(与行块无关,每 program 读同一 A[K])
    scrA = tle.spread(A, cube_shape=(KC, CN, CK))  # memref<KC×64>
    acc = tle.vzero(f32, group=B1 * B2)  # <8×64xf32>
    for kc in tle.range(0, KC, 1):
        vb = tle.vload(Bcube, (0, kc), group=B1)  # <2×64xf16>
        va1 = tle.vload(scrA, (kc, 0))  # <64xf16>(单 cube,A 已 n 广播)
        va = tle.vbroadcast(va1, B2)  # <4×64xf16> 寄存器复制 b2 份
        acc = tle.vmadot(acc, vb, va)  # cross_batch_matmul
    # ⑤ 输出还原:vpack(vector) 逐级 group_interleave → 行主序
    c1 = tle.vpack(acc, 8)  # <8×64> → <4×128>
    c2 = tle.vpack(c1, 16)  # <4×128> → <2×256>
    cf = tle.vshape(c2, (MB, Npad))  # 行主序 <MB×Npad>
    c = tle.cast(cf, f16)
    tle.vstore(C, row_base * Npad, c, shape=(MB, Npad))  # 写回本块


@triton.jit
def host(B, A, C, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    _sr_call(mv, outputs=[], inputs=[B, A, C, row_base])


def test_mv_cbm():
    rng = np.random.default_rng(0)
    Blog = rng.standard_normal((M, K)).astype(np.float16)
    Alog = rng.standard_normal((K,)).astype(np.float16)
    golden = Blog.astype(np.float64) @ Alog.astype(np.float64)  # [M]
    B = torch.tensor(Blog.reshape(-1))  # 行主序,DSL 内 vpack 摆 cube
    A = torch.tensor(Alog.reshape(-1))  # 原始 <K> 向量,DSL 内 spread 广播,零 host 预处理
    C = torch.zeros(M, Npad, dtype=torch.float16)
    grid = (M // MB, )                  # M//MB 个 program(grid 并发)
    host[grid](B.contiguous(), A.contiguous(), C, BLOCK=MB)
    got = C[:, 0].float().numpy().astype(np.float64)  # mv 结果在每行 col0
    diff = np.abs(got - golden).max()
    assert diff < 5e-2, f"mv max_diff={diff:.4e}\ngot={got[:4]}\ngold={golden[:4]}"
