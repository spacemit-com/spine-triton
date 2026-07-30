"""spine_raw layernorm — (x - mean) * rsqrt(var + eps)

三趟 reduce：
  趟1: sum(x)/N  → mean (scalar)
  趟2: sum((x-mean)²)/N → var (scalar), 每 tile 里 vec-scalar broadcast 减均值
  趟3: (x-mean)*rsqrt(var+eps) 回写

验证 L0 scalar ÷ index + 1D full-vector vstore + vec-scalar broadcast 都通。
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

EPS = 1e-5


# ---------------------------------------------------------------------------
# layernorm_1d: C[i] = (x[i] - mean) / sqrt(var + eps)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def layernorm_1d_kernel(X: tle.mem(f16), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # ── 趟1: sum(x) ──────────────────────────────────────────────────────
    acc1 = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        va = tle.cast(tle.vload(X, i), f32)
        acc1 = acc1 + va
    for i in tle.range(Nfloor, N, nvl):
        nvl_t1 = tle.vconfig(N - i, 1)
        ta = tle.cast(tle.vload(X, i), f32)
        acc1 = acc1 + ta
    mean = tle.vreduce_sum(acc1) / N          # f32 scalar

    # ── 趟2: sum(x²) — use E[x²]-mean² to avoid (0-mean)² inflation from padding ─────
    acc2 = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vb = tle.cast(tle.vload(X, i), f32)
        acc2 = acc2 + vb * vb
    for i in tle.range(Nfloor, N, nvl):
        nvl_t2 = tle.vconfig(N - i, 1)
        tb = tle.cast(tle.vload(X, i), f32)   # fill=0: 0²=0, no inflation
        acc2 = acc2 + tb * tb
    var = tle.vreduce_sum(acc2) / N - mean * mean   # E[x²] - mean² = Var(x)
    scale = tle.rsqrt(var + EPS)              # f32 scalar (f32 + f32 literal)

    # ── 趟3: (x - mean) * scale ───────────────────────────────────────────
    for i in tle.range(0, Nfloor, nvl):
        vc = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, (vc - mean) * scale)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t3 = tle.vconfig(N - i, 1)
        tc = tle.cast(tle.vload(X, i), f32)
        tle.vstore(out, i, (tc - mean) * scale)


@triton.jit
def layernorm_1d_host(X, out, N):
    _sr_call(layernorm_1d_kernel, outputs=[], inputs=[X, out, N])


def _ref_layernorm(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    mean = xf.mean()
    var = ((xf - mean) ** 2).mean()
    return (xf - mean) / torch.sqrt(var + EPS)


@pytest.mark.parametrize("N", [64, 128, 256, 512])
def test_layernorm_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float16)
    out = torch.zeros(N, dtype=torch.float32)
    layernorm_1d_host[(1,)](X, out, N)
    ref = _ref_layernorm(X)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 任意 N（非 VL 整倍数）——验证尾部 pad 处理
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [100, 200, 300])
def test_layernorm_1d_arb(N):
    torch.manual_seed(7)
    X = torch.randn(N, dtype=torch.float16)
    out = torch.zeros(N, dtype=torch.float32)
    layernorm_1d_host[(1,)](X, out, N)
    ref = _ref_layernorm(X)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
