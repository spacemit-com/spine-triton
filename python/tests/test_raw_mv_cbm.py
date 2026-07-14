"""PLAN §4.2 mv:纯 raw eDSL mv(cbm 矩阵引擎),参数化到「任意 8 的倍数 shape 族」。

mv: C[M] = B[M,K] @ A[K]。表达成 GEMM,A 是被广播维(cube 的 n 维退化)。
- B 侧:vpack(memref)→linalg.pack 摆 cube。
- A 侧:tle.spread 标量广播成 cube scratch(绕 vscale)+ vbroadcast 寄存器复制 b2 份。
- 输出:vpack(vector) 逐级 group_interleave 还原行主序,取 col0。

shape 支持(本文件验证的能力边界):
- **M:任意 % MB==0(MB=16)** —— 每 program 固定算一个 16 行 cube 块(=已坐实的还原链),
  M 靠 grid=(M//MB,) 多起 program 扩,还原链不变。
- **K:任意 % CK==0(CK=8)** —— K 只驱动 KC=K//CK 归约循环 + B pack stride;
  输出 C 是 <M×Npad> 与 K 无关,还原链不变。
- **Npad 固定 32**(A 的广播维,mv 真结果只在 col0)。更大 Npad 需 N 方向 tiling:
  vpack.vv 的 seg=groupLen×bitwidth≤512(f32 acc)→ b2≤4 → Npad≤32,是硬件 VPACK_TYPE
  上限,不是还原链长度问题(SPEC §6.3)。
- 非 8 整除的尾部(M%16≠0 / K%8≠0)需 padding 或 mask(SPEC §6.1 avl 收窄,待扩),本文件不覆盖。

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

CM, CN, CK = tle.mma_cube(f16)  # (8, 8, 8) for f16
VL = CN * CK                    # cube lane 宽 = n×k,f16→64
MB = 2 * CM                     # 每 program 的 cube 行块 = 16(b1=MB/CM=2,匹配已坐实还原链)
Npad = 4 * CN                   # A 广播维 = 32(b2=Npad/CN=4,seg=2*CN*16=256≤512 硬件上限)
B1, B2 = MB // CM, Npad // CN   # 2, 4


def make_mv(M, K):
    """生成 (kernel, host) —— C[M]=B[M,K]@A[K]。

    K 任意(PLAN_pad §2.1/§2.2:vpack fill+insert 补 K 尾 + spread k_real 补 K 尾)。
    M:整除 MB 时 grid=M/MB 多 program;M<MB 时 grid=1 单 block(vpack fill+insert 补 M 尾,
    对齐 probe_fill_pad)。M>MB 且非整除的 per-program 动态行数属 §2.B 待扩,本函数不覆盖。
    每 shape 唯一 __name__ 避免 Triton JIT 按名缓存串用。
    """
    Kp = ((K + CK - 1) // CK) * CK      # K 上取整到 CK 倍数(pad 尾补 0)
    KC = Kp // CK                        # K 循环迭代数(按 pad 后)
    if M % MB == 0:
        grid_blocks = M // MB
        MROWS = MB                        # 每 program 满 MB 行(整除)
    else:
        assert M < MB, f"non-divisible M>{MB} needs §2.B dynamic rows (未实现); got M={M}"
        grid_blocks = 1
        MROWS = M                         # 单 block:vpack 读真实 M 行, codegen fill+insert 补到 MB

    @tle.raw_kernel
    def mv(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f16, out=True), row_base: tle.index):
        nvl = tle.vconfig(VL, 1)  # 活跃 VL = cube lane 宽(经 _active_vl 供 vzero/vload)
        # ① B 侧:vpack(memref) 读真实 MROWS 行;M/K 非整除时 codegen 自动 fill+insert 补到 MB/Kp
        Bcube = tle.vpack(B, inner_tiles=(MB, CK), stride=K, rows=MROWS, offset=row_base * K)
        # ② A 侧:spread 标量广播成 cube scratch;k_real=K 让尾 tile 填 0(避免 over-read A)
        scrA = tle.spread(A, cube_shape=(KC, CN, CK), k_real=K)
        acc = tle.vzero(f32, group=B1 * B2)  # <8×64xf32>
        for kc in tle.range(0, KC, 1):
            vb = tle.vload(Bcube, (0, kc), group=B1)   # <2×64xf16>
            va1 = tle.vload(scrA, (kc, 0))             # <64xf16>(单 cube,A 已 n 广播)
            va = tle.vbroadcast(va1, B2)               # <4×64xf16>
            acc = tle.vmadot(acc, vb, va)              # cross_batch_matmul
        # ⑤ 输出还原:group_interleave 逐级(groupLen 从 CN 起翻倍),cube→行主序 <MB×Npad>
        c1 = tle.vpack(acc, CN)       # <8×64> → <4×128>
        c2 = tle.vpack(c1, 2 * CN)    # <4×128> → <2×256>
        cf = tle.vshape(c2, (MB, Npad))
        c = tle.cast(cf, f16)
        tle.vstore(C, row_base * Npad, c, shape=(MB, Npad))

    mv._fn.__name__ = f"mv_cbm_{M}_{K}"

    @triton.jit
    def host(B, A, C, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        _sr_call(mv, outputs=[], inputs=[B, A, C, row_base])

    host.__name__ = f"_mv_cbm_host_{M}_{K}"
    host.fn.__name__ = host.__name__
    host._grid_blocks = grid_blocks
    return host


def _run(M, K):
    rng = np.random.default_rng(0)
    Blog = rng.standard_normal((M, K)).astype(np.float16)
    Alog = rng.standard_normal((K,)).astype(np.float16)
    golden = Blog.astype(np.float64) @ Alog.astype(np.float64)
    B = torch.tensor(Blog.reshape(-1))
    A = torch.tensor(Alog.reshape(-1))
    Mp = ((M + MB - 1) // MB) * MB       # kernel 每 block 写 MB 行 → C 按 Mp 分配, 尾行丢弃
    C = torch.zeros(Mp, Npad, dtype=torch.float16)
    host = make_mv(M, K)
    host[(host._grid_blocks,)](B.contiguous(), A.contiguous(), C, BLOCK=MB)
    got = C[:M, 0].float().numpy().astype(np.float64)   # 取真实 M 行 col0
    diff = np.abs(got - golden).max()
    assert diff < 5e-2, f"M={M} K={K} max_diff={diff:.4e}\ngot={got[:4]}\ngold={golden[:4]}"


# 整除族(回归)+ 任意 shape(K 非整除 / M<MB 非整除)
_SHAPES = [(64, 64), (128, 64), (64, 128), (256, 64),    # 整除回归
           (64, 60), (128, 100), (64, 40),               # K 非整除(fill+insert 补 K 尾)
           (12, 60), (12, 64), (4, 40)]                  # M<MB 且 K 任意(单 block fill+insert 补 M/K)


@pytest.mark.parametrize("M, K", _SHAPES)
def test_mv_cbm(M, K):
    _run(M, K)
