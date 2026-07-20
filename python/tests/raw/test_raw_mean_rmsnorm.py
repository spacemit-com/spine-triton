"""spine_raw L0 标量算术验证 — reduce 后 /N + rsqrt 广播回向量。

验证 PLAN_reduce_gap.md L0 修复:codegen 支持 f32 scalar 与 index 混合算术
(`vreduce_sum(v) / N`),解锁 mean / rms_norm 家族。
"""
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
# mean_1d: sum(x) / N  —— 最小 L0 标量除法验证
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mean_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.cast(tle.vload(X, i), f32)
        acc = acc + vx
    for i in tle.range(Nfloor, N, nvl):
        nvl_tail = tle.vconfig(N - i, 1)
        tx = tle.cast(tle.vload(X, i), f32)
        acc = acc + tx
    s = tle.vreduce_sum(acc)
    tle.sstore(out, 0, s / N)      # ← L0: f32 scalar / index


@triton.jit
def mean_1d_host(X, out, N):
    _sr_call(mean_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 100, 257])
def test_mean_1d(N):
    X = torch.randn(N, dtype=torch.float16)
    out = torch.zeros(1, dtype=torch.float32)
    mean_1d_host[(1,)](X, out, N)
    ref = X.to(torch.float32).mean()
    torch.testing.assert_close(out[0], ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# rms_norm_1d: x / sqrt(mean(x^2) + eps)  —— 完整 L0 下游链
#   reduce → /N → rsqrt(scalar) → 标量广播回向量 mul
# ---------------------------------------------------------------------------
@tle.raw_kernel
def rms_norm_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.cast(tle.vload(X, i), f32)
        acc = acc + vx * vx
    for i in tle.range(Nfloor, N, nvl):
        nvl_tail = tle.vconfig(N - i, 1)
        tx = tle.cast(tle.vload(X, i), f32)
        acc = acc + tx * tx
    ms = tle.vreduce_sum(acc) / N          # mean of squares (scalar)
    scale = tle.rsqrt(ms)                   # rsqrt on scalar
    # 归一化循环用独立临时名(nx/mx),避免与 reduce 循环的 vx/tx 同名 →
    # _find_reassigned 会把出作用域的 vx/tx 误当 iter_arg → 引用子 region SSA。
    for i in tle.range(0, Nfloor, nvl):
        nx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, nx * scale)      # scalar broadcast into vector
    for i in tle.range(Nfloor, N, nvl):
        nvl_tail2 = tle.vconfig(N - i, 1)
        mx = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, mx * scale)


@triton.jit
def rms_norm_1d_host(X, out, N):
    _sr_call(rms_norm_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256])
def test_rms_norm_1d(N):
    X = torch.randn(N, dtype=torch.float16)
    out = torch.zeros(N, dtype=torch.float32)
    rms_norm_1d_host[(1,)](X, out, N)
    xf = X.to(torch.float32)
    ref = xf / torch.sqrt((xf * xf).mean())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
