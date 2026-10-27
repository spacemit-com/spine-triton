"""smt_rvisa softmax — stable numerics via max-subtracted exp-sum.

Kernel: out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))

Uses: vreduce_max + vexp + vreduce_sum (all L1 primitives).
"""
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f32 = sr.f32


@sr.raw_kernel
def softmax_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # ── 趟1: max(x) ───────────────────────────────────────────────────────
    acc_max = sr.vload(X, 0, dtype=f32)  # seed with first tile
    for i in sr.range(0, Nfloor, nvl):
        va = sr.vload(X, i, dtype=f32)
        acc_max = sr.vmax(acc_max, va)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        ta = sr.vload(X, i, dtype=f32)
        acc_max = sr.vmax(acc_max, ta)
    xmax = sr.vreduce_max(acc_max)  # scalar

    # ── 趟2: sum(exp(x - max)) ────────────────────────────────────────────
    acc_sum = sr.vzero(f32)
    for i in sr.range(0, Nfloor, nvl):
        vb = sr.vload(X, i, dtype=f32)
        acc_sum = acc_sum + sr.vexp(vb - xmax)  # vexp on vec-scalar sub
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        # fill=-1e38: padded lanes get exp(-1e38 - xmax)≈0, don't inflate denom
        tb = sr.vload(X, i, dtype=f32, fill=-1e38)
        acc_sum = acc_sum + sr.vexp(tb - xmax)
    denom = sr.vreduce_sum(acc_sum)  # scalar

    # ── 趟3: exp(x - max) / denom ─────────────────────────────────────────
    inv_denom = 1.0 / denom  # f32 ÷ f32 scalar (L0)
    for i in sr.range(0, Nfloor, nvl):
        vc = sr.vload(X, i, dtype=f32)
        sr.vstore(out, i, sr.vexp(vc - xmax) * inv_denom)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tc = sr.vload(X, i, dtype=f32)
        sr.vstore(out, i, sr.vexp(tc - xmax) * inv_denom)


@triton.jit
def softmax_1d_host(X, out, N):
    _sr_call(softmax_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256])
def test_softmax_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    softmax_1d_host[(1, )](X, out, N)
    ref = torch.softmax(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("N", [100, 200])
def test_softmax_1d_arb(N):
    torch.manual_seed(7)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    softmax_1d_host[(1, )](X, out, N)
    ref = torch.softmax(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)
